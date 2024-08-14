
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy
# from sklearn.manifold import TSNE
from config import Config_MBM_EEG
<<<<<<< HEAD
from eegtoimage.dataset import EEGDataset_subject, eeg_pretrain_dataset
from sc_mbm.mae_for_eeg_2 import MAEforEEG
from sc_mbm.trainer import train_one_epoch, validate
from sc_mbm.trainer import EarlyStopping
=======
from dataset import EEGDataset_subject
from sc_mbm.mae_for_eeg_2 import MAEforEEG
from sc_mbm.trainer import train_one_epoch, validate
# from sc_mbm.trainer import EarlyStopping
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model    


os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init(
<<<<<<< HEAD
                    project="dreamdiffusion exp1",
=======
                    project="dreamdiffusion",
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
                    anonymous="allow",
                    group='stageA_sc-mbm',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MBM pre-training for fMRI', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--decoder_num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)

    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)

    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
<<<<<<< HEAD

    # (single / multi) object
    parser.add_argument('--subject', type =str, default="multi")
=======
                        
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def main(config):
    print('num of gpu:')
    print(torch.cuda.device_count())

    # 여기서는 local_rank가 gpu 번호를 말함.
    local_rank = config.local_rank

    # Initialize process group for distributed training
<<<<<<< HEAD
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
=======
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.distributed.init_process_group(backend='nccl')
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
    
    config.local_rank = local_rank
    output_path = os.path.join(config.root_path, 'results', 'eeg_pretrain',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    logger = wandb_logger(config) if local_rank == 0 else None
    
    if local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

<<<<<<< HEAD
    # early_stopper = EarlyStopping(patience=10)

# # multi subject dataloader
    train_dataset = EEGDataset_subject(eeg_signals_path="/Data/summer24/DreamDiffusion/datasets/eeg_data/train/train_dataset.pth", mode = "train")
=======
    # early_stopper = EarlyStopping(patience=5)

    train_dataset = EEGDataset_subject(eeg_signals_path="/Data/summer24/DreamDiffusion/datasets/eegdata_2subject/train.pth", mode = "train")
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, rank=local_rank)
    train_dataloader_eeg = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, 
                shuffle=False, pin_memory=True)

<<<<<<< HEAD
    valid_dataset = EEGDataset_subject(eeg_signals_path='/Data/summer24/DreamDiffusion/datasets/eeg_data/valid/val_dataset.pth', mode = "val")
=======
    valid_dataset = EEGDataset_subject(eeg_signals_path='/Data/summer24/DreamDiffusion/datasets/eegdata_2subject/val.pth', mode = "val")
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873

    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, rank=local_rank)
    valid_dataloader_eeg = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=valid_sampler, 
                shuffle=False, pin_memory=True)
   
    print(f'Dataset size: {len(train_dataset)}\n Time len: {train_dataset.data_len}')


<<<<<<< HEAD
# single subject (number 2) dataloader

    # train_dataset = EEGDataset_subject(eeg_signals_path="/Data/summer24/DreamDiffusion/data/eegdata_2subject/train.pth", mode = "train")

    # train_sampler = torch.utils.data.DistributedSampler(train_dataset, rank=local_rank)
    # train_dataloader_eeg = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, 
    #             shuffle=False, pin_memory=True)

    # valid_dataset = EEGDataset_subject(eeg_signals_path='/Data/summer24/DreamDiffusion/data/eegdata_2subject/val.pth', mode = "val")

    # valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, rank=local_rank)
    # valid_dataloader_eeg = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=valid_sampler, 
    #             shuffle=False, pin_memory=True)
   
    # print(f'Dataset size: {len(train_dataset)}\n Time len: {train_dataset.data_len}')

    
=======
    # create model
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
    config.time_len=train_dataset.data_len
    model = MAEforEEG(time_len=train_dataset.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,
                    decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                    num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                    focus_range=config.focus_range, focus_rate=config.focus_rate, 
<<<<<<< HEAD
                    img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss, num_classes=config.num_classes)
=======
                    img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss, num_classes=config.num_classes)   
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
    
    model.to(device)
    model_without_ddp = model
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
<<<<<<< HEAD
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
=======
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=config.use_nature_img_loss)
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873

    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model, log='all', log_freq=1000)

    f1_score_list = []
    start_time = time.time()
    print('Start Training the EEG MAE ... ...')
    img_feature_extractor = None
    preprocess = None
    if config.use_nature_img_loss:
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        m = resnet50(weights=weights)   
        img_feature_extractor = create_feature_extractor(m, return_nodes={f'layer2': 'layer2'}).to(device).eval()
        for param in img_feature_extractor.parameters():
            param.requires_grad = False

    for ep in range(config.num_epoch):
        
        if torch.cuda.device_count() > 1: 
            train_sampler.set_epoch(ep) # to shuffle the data at every epoch
        f1_score = train_one_epoch(model, train_dataloader_eeg, optimizer, device, ep, loss_scaler, logger, config, start_time, model_without_ddp,
                            img_feature_extractor, preprocess)
        f1_score_list.append(f1_score)
        # if (ep % 20 == 0 or ep + 1 == config.num_epoch) and local_rank == 0: #and ep != 0
            # save models
        # if True:
<<<<<<< HEAD
        # plot figures
        plot_recon_figures(model, device, train_dataset, output_path, 5, config, logger, model_without_ddp)

        val_loss, val_f1, val_acc = validate(model, valid_dataloader_eeg, device, config)
        save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path, 'checkpoints'), val_acc)
=======
        save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
        # plot figures
        plot_recon_figures(model, device, train_dataset, output_path, 5, config, logger, model_without_ddp)

        val_loss, val_f1 = validate(model, valid_dataloader_eeg, device, config)
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873

        # if early_stopper.should_stop(model,val_loss):
        #     print(f"EarlyStopping: [Epoch: {ep - early_stopper.counter}]")
        #     break

        if logger is not None:
            logger.log('val_loss', val_loss, step=ep)
            logger.log('val_f1', val_f1, step=ep)
<<<<<<< HEAD
            logger.log('val_acc', val_acc, step = ep)
=======
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
<<<<<<< HEAD
=======
        # logger.log('max cor', np.max(cor_list), step=config.num_epoch-1)
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
        logger.finish()
    return


@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['eeg']
        sample = sample.to(device)
        _, pred, mask,output = model(sample, mask_ratio=config.mask_ratio)
<<<<<<< HEAD
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
=======
        # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
<<<<<<< HEAD
        ax[0].plot(x_axis, sample)
=======
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
>>>>>>> f83336c493fc7ab4824f264a2cab7599cd95f873
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)



def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_EEG()
    config = update_config(args, config)
    main(config)