import os
import numpy as np
import torch

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 
class Config_MBM_fMRI(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 100
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 16
        self.embed_dim = 1024 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

class Config_MBM_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.dataset = 'GOD' # GOD  or BOLD5000
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.include_nonavg_test = True
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI1']

        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.75 
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.

        # distributed training
        self.local_rank = 0,1
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

class Config_MBM_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.dataset = 'GOD' # GOD  or BOLD5000
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.include_nonavg_test = True
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI1']

        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.75 
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.

        # distributed training
        self.local_rank = 0
        

# 수정한 코드
class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.root_path = '../DreamDiffusion/'
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.crop_ratio = 0.2
        self.lr = 1e-5
        self.min_lr = 0.
        self.weight_decay = 0.15
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 128
        self.clip_grad = 0.8
        self.num_classes = 40

        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 4 #  1s

        self.embed_dim = 1024 #256 # has to be a multiple of num_heads -> 원래 dimension은 128차원 -> num_heads =8 -> 128*8 = 1024
        self.decoder_embed_dim = 512 #128
        self.depth = 12
        self.num_heads = 8
        self.decoder_num_heads = 8
        self.mlp_ratio = 1.0
        self.img_size = 512

        # Project setting
        self.root_path = 'DreamDiffuion/'
        self.output_path = 'DreamDiffuion/output'
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0



class Config_EEG_finetune(Config_MBM_finetune):
    def __init__(self):
      
        # Project setting
        self.root_path = '../DreamDiffusion/'
        self.output_path = '../DreamDiffusion/output'

        self.dataset = 'EEG'

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')

        self.dataset = 'EEG' 

        self.pretrain_mbm_path = 'DreamDiffuion/results/eeg_pretrain/11-07-2024-07-07-28/checkpoints/checkpoint.pth'
        self.include_nonavg_test = True


        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 128 if self.dataset == 'GOD' else 128
        self.mask_ratio = 0.5
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0
        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = './DreamDiffusion/'
        self.output_path = './DreamDiffusion/exps/'
        self.eeg_signals_path = os.path.join(self.root_path, 'data/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'data/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = None

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = '/Data/summer24/DreamDiffusion/stage1_weight/eegfeat_all_0.9702620967741935.pth'

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        # memeory 문제로 batch 5->1
        self.batch_size = 8
        # self.lr = 5.3e-5
        self.lr = 1e-4
        self.num_epoch = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 6
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 

class Config_Cls_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = './DreamDiffusion/'
        self.output_path = './DreamDiffusion/exps/'

        self.eeg_signals_path = os.path.join('/Data/summer24/DreamDiffusion/datasets/eeg_5_95_std.pth')
        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_14_70_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        self.splits_path = os.path.join('/Data/summer24/DreamDiffusion/datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 4 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 10
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 6
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None
        self.temperature = 1.5