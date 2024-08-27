# stage 2 Fine-tuning with limited EEG image pairs & Align EEG, text and image spaces

import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping 
from sc_mbm.network import EEGFeatNet
from PIL import Image



def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class Projection(nn.Module):
    def __init__(self, input_dim):
        super(Projection, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x_proj = self.linear(x.transpose(1,2))
        x_proj = x_proj.squeeze(-1)
        return x_proj

class cond_stage_model(nn.Module):
    # cond_dim=768
    def __init__(self, metafile, num_voxels=400, cond_dim=768, global_pool=False, clip_tune = True, cls_tune = False):
        super().__init__()
        # prepare pretrained eeg_encoder
        model = EEGFeatNet(n_classes=40, in_channels=128, n_features=128, projection_dim=128, num_layers=1)
        model.load_checkpoint(metafile['model_state_dict'])

        self.encoder = model

        if clip_tune:
            self.projection_layer = Projection(input_dim=77)
        if cls_tune:
            self.cls_net = classify_network()
        # self.fmri_seq_len = model.num_patches -> (time_len//patch_size) 1024
        # self.fmri_latent_dim = model.embed_dim -> 1024
        self.fmri_seq_len = 128
        self.fmri_latent_dim = 440

        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )

        # if global_pool == False:
        # self.channel_mapper = nn.Sequential(
        #     nn.Conv1d(128, 64, 1, bias=True),
        #     nn.Conv1d(64, 77, 1, bias=True)
        # )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

    def forward(self, x):
        # n, c, w = x.shape
        # latent_crossattn = [3,440,128]
        latent_crossattn = self.encoder(x)
        # latent_return = [3,77,440]
        latent_return = latent_crossattn.transpose(1,2)
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_return)
        out = self.dim_mapper(latent_crossattn)
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        # target_emb = self.mapping(x)

        target_emb = self.projection_layer(x)     
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)
        loss = 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss
    



class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
    # taetae edit
    # def __init__(self, metafile, num_voxels, device=torch.device('cuda'),
            pretrain_root='../pretrains/',
            logger=None, ddim_steps=125, global_pool = True, use_time_cond=False, clip_tune = True, cls_tune = False, temperature=1.0):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')

        self.ckp_path = '/Data/summer24/DreamDiffusion/pretrains/models/v1-5-pruned.ckpt'
        self.config_path = os.path.join('/Data/summer24/DreamDiffusion/pretrains/models/config15.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool
        self.cond_dim = config.model.params.unet_config.params.context_dim
        # print(config.model.target)

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True

        # model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune)
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune, cls_tune = cls_tune)
        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device
        self.model = model

        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile
        self.temperature=temperature

    def finetune(self, trainers, dataset, valid_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one

        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        print(f'batch_size is: {bs1}')

        
        train_loader = DataLoader(dataset, batch_size=bs1,pin_memory=False, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=bs1,pin_memory=False, shuffle=True)

        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        #  create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=2)
        trainers.fit(self.model, train_loader, val_dataloaders=valid_loader)

        self.model.unfreeze_whole_model()

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()
            },
            os.path.join(output_path, 'checkpoint_eLDM.pth')
        )


    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None, shouldSave = True):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model, temperature=self.temperature)
        # sampler = DDIMSampler(model)
        if state is not None:
            # taetae
            # torch.cuda.set_rng_state(state)
            torch.set_rng_state(state)

        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                latent = item[0]
                gt_image = rearrange(item[1], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'

                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)

                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
                
                if output_path is not None and shouldSave == True:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path,
                            f'./test{count}-{copy_idx}.png'))

        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')

        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)