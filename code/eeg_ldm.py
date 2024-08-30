import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy
from tqdm import tqdm
import time
from torch.nn import Identity
import lpips
import config2
from torch.utils.data import DataLoader
from dataset import EEGDataset
from config import Config_Generative_Model
from dc_ldm.ldm_for_eeg import eLDM
from eval_metrics import get_similarity_metric
from natsort import natsorted
import os
import cv2



def wandb_init(config, output_path):
    wandb.init( project='eeg_diffusion',
                group="stageB_dc-ldm",
                anonymous="allow",
                config=config,
                reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []
    
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None, 
                        n_way=40, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list
               
def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples, 
                config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path, 
                            f'./test{sp_idx}-{copy_idx}.png'))

    wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    wandb.log(metric_dict)

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    config = Config_Generative_Model()
    start_time = time.time()
    base_path       = config2.base_path
    train_path      = config2.train_path
    validation_path = config2.validation_path
    device          = config2.device

    x_train_eeg = []
    x_train_image = []
    x_train_raw_img = []
    labels = []

    batch_size     = config2.batch_size
    EPOCHS         = config2.epoch
    num_voxels = config2.num_voxels
    class_labels   = {}
    label_count    = 0

# 이 부분이 npy로 되어 있는 train / val / test 로 나눠진 데이터들을 불러오는 과정
    for i in tqdm(natsorted(os.listdir(base_path + train_path))):
        loaded_array = np.load(base_path + train_path + i, allow_pickle=True)
        x_train_eeg.append(loaded_array[1].T)
        x_train_raw_img.append(loaded_array[0])
        img = cv2.resize(loaded_array[0], (512, 512))
        img =((img) - 127.5) / 127.5 
        img = np.transpose(img, (2, 0, 1))
        x_train_image.append(img)
        labels.append(loaded_array[2])
        
    x_train_eeg   = np.array(x_train_eeg)
    x_train_image = np.array(x_train_image)
    train_labels  = np.array(labels)

    # ## convert numpy array to tensor
    x_train_eeg   = torch.from_numpy(x_train_eeg).float().to(device)
    x_train_image = torch.from_numpy(x_train_image).float().to(device)
    train_labels  = torch.from_numpy(train_labels).long().to(device)

    train_data       = EEGDataset(x_train_eeg, x_train_image, x_train_raw_img, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)

    # my loader
    # img.shape [3,512,512]
    # train_data[0][0] (1 sample) eeg [440,128] / img [3,512,512]


    x_val_eeg = []
    x_val_image = []
    x_val_raw_img = []
    label_Val = []

    for i in tqdm(natsorted(os.listdir(base_path + validation_path))):
        loaded_array = np.load(base_path + validation_path + i, allow_pickle=True)
        x_val_eeg.append(loaded_array[1].T)
        x_val_raw_img.append(loaded_array[0])
        img = cv2.resize(loaded_array[0], (512, 512))
        img =((img) - 127.5) / 127.5 
        img = np.transpose(img, (2, 0, 1))
        x_val_image.append(img)
        label_Val.append(loaded_array[2])
        
    x_val_eeg   = np.array(x_val_eeg)
    x_val_image = np.array(x_val_image)
    val_labels  = np.array(label_Val)

    x_val_eeg   = torch.from_numpy(x_val_eeg).float().to(device)
    x_val_image = torch.from_numpy(x_val_image).float().to(device)
    val_labels  = torch.from_numpy(val_labels).long().to(device)

    val_data       = EEGDataset(x_val_eeg, x_val_image,x_val_raw_img, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    # prepare pretrained mbm 
    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')

    # create generateive model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger, 
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune = config.clip_tune, cls_tune = config.cls_tune)
    

    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('model resumed')
    # finetune the model
    trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=2)
    generative_model.finetune(trainer, train_data, val_data,
                config.batch_size, config.lr, config.output_path, config=config)

    # generate images
    generate_images(generative_model, train_data, val_data, config)

    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default = '../dreamdiffusion/')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=0):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    # return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger, 
    #         precision=precision, accumulate_grad_batches=accumulate_grad_batches,
    #         enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
    #         check_val_every_n_epoch=check_val_every_n_epoch, devices=8, strategy = 'ddp')
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger, 
        precision=precision, accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
        check_val_every_n_epoch=check_val_every_n_epoch)
  

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)
    
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    output_path = os.path.join(config.output_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    wandb_init(config, output_path)

    main(config)