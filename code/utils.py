
from einops import rearrange
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import pytorch_lightning as pl
import numpy as np
from eval_metrics import get_similarity_metric


# def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=10):
#     acc = 'gpu' if torch.cuda.is_available() else 'cpu'
#     return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger,
#             precision=precision, accumulate_grad_batches=accumulate_grad_batches,
#             enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
#             check_val_every_n_epoch=check_val_every_n_epoch, limit_val_batches=0.15, limit_test_batches=0.15, limit_predict_batches=0.5, devices=8, strategy="ddp")


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=10):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger,
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, gradient_clip_val=0.5,
            check_val_every_n_epoch=check_val_every_n_epoch, limit_val_batches=0.15, limit_test_batches=0.15, limit_predict_batches=0.5)


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


    # No class metric for now
    # res_part = []
    # for s in samples_to_run:
    #     pred_images = [img[s] for img in samples]
    #     pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
    #     res = get_similarity_metric(pred_images, gt_images, 'class', None,
    #                     n_way=50, num_trials=50, top_k=1, device='cuda')
    #     res_part.append(np.mean(res))
    # res_list.append(np.mean(res_part))
    # res_list.append(np.max(res_part))
    # metric_list.append('top-1-class')
    # metric_list.append('top-1-class (max)')
    return res_list, metric_list

def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 3) # generate 3
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples,
                config.ddim_steps, config.HW, 3)
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