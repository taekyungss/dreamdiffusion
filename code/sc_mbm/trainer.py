import math, sys
import torch
import sc_mbm.utils as ut
# from torch._six import inf
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
from sc_mbm.mae_for_eeg_2 import freeze_weights, unfreeze_weights



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def unpatchify(self, x):
    """
    x: (N, L, patch_size)
    imgs: (N, 1, num_voxels)
    """
    p = self.patch_embed.patch_size
    h = x.shape[1]

    imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
    return imgs.transpose(1,2)




# def compute_f1_score(output, labels):
#     _, preds = torch.max(output, 1)
#     preds = preds.cpu().numpy()
#     labels = labels.cpu().numpy()
#     return f1_score(labels, preds, average='weighted')


def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                        loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []

    total_f1=[]
    accum_iter = config.accum_iter

    for data_iter_step, (data_dcit) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        samples = data_dcit['eeg']
        labels = data_dcit["label"]
        
        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = data_dcit['image']
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        
        samples = samples.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()

        # with torch.cuda.amp.autocast(enabled=True):
        loss, pred, _ , output = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)
        loss = loss + criterion(output, labels.long())

        loss_value = loss.item()



        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        
        f1 = f1_score(labels, preds, average='weighted')

        total_f1.append(f1)
        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)

        # definiton of cor : 상관계수 계산 -> 우리가 하고 있는게 eeg data를 masking해서 원본 신호처럼 강건하게 reconstruction 하는거니까!
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        
        optimizer.zero_grad()

        total_loss.append(loss_value)
        # total_cor.append(cor)

        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)

        log_writer.log('cor', np.mean(total_cor), step=epoch)

        log_writer.log('f1_score', np.mean(total_f1), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)


    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] train_loss: {np.mean(total_loss)}, train_f1_score: {np.mean(total_f1)}')

    return np.mean(total_f1)




def validate(model, dataloader, device, config, model_without_ddp=None, log_writer=None):
    model.eval()
    total_loss = []
    total_f1 = []
    total_acc = []

    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            sample = batch['eeg'].to(device)
            labels = batch['label'].to(device)
            criterion = nn.CrossEntropyLoss()

            with torch.cuda.amp.autocast(enabled=True):
                loss, pred, _ , output = model(sample, mask_ratio=config.mask_ratio)
                loss = loss + criterion(output, labels.long())
            loss_value = loss.item()
        
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            f1 = f1_score(preds, labels, average='weighted')
            acc = accuracy_score(preds, labels)

            total_f1.append(f1)
            total_acc.append(acc)
            num_samples += 1

            print('valid_loss_step:', loss_value)

        if log_writer is not None:
            log_writer.log('valid_loss_step', np.mean(total_loss), step=num_samples)
            log_writer.log('f1_score', np.mean(total_f1), step=num_samples)
            log_writer.log('accuracy', np.mean(total_acc), step=num_samples)
        total_loss.append(loss_value)

    avg_loss = np.mean(total_loss)
    max_f1 = np.max(total_f1)
    max_acc = np.max(total_acc)

    if config.local_rank == 0:        
        print(f'valid_loss_step: {avg_loss}, val_f1_score: {max_f1}, max_accuracy : {max_acc}')
    return avg_loss, max_f1, max_acc


class EarlyStopping(object):
    def __init__(self, patience=2, save_path="model.pth"):
        self._min_loss = np.inf
        self._patience = patience
        self._path = save_path
        self.__counter = 0
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            torch.save(model.state_dict(), self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False
   
    def load(self, model):
        model.load_state_dict(torch.load(self._path))
        return model
    
    @property
    def counter(self):
        return self.__counter