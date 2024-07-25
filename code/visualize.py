from sc_mbm.mae_for_eeg import MAEforEEG
from config import Config_MBM_EEG
from dataset import eeg_pretrain_dataset
from stageA1_eeg_pretrain import fmri_transform
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from sklearn.manifold import TSNE
import numpy as np

import torch
import torch.nn as nn

config = Config_MBM_EEG()
local_rank = config.local_rank
device = torch.device(f'cuda:{local_rank}')

valid_dataset = eeg_pretrain_dataset(path='/Data/summer24/DreamDiffusion/datasets/eegdata/train/eeg', roi=config.roi, patch_size=config.patch_size,
                transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit, 
            include_kam=config.include_kam, include_hcp=config.include_hcp)

valid_dataloader_eeg = DataLoader(valid_dataset, batch_size=config.batch_size,
            shuffle=False, pin_memory=True)


model = MAEforEEG(time_len=1024, patch_size=config.patch_size, embed_dim=config.embed_dim,
                    decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                    num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                    focus_range=config.focus_range, focus_rate=config.focus_rate, 
                    img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss)

checkpoint = torch.load("/Data/summer24/DreamDiffusion/DreamDiffuion/results/eeg_pretrain/11-07-2024-07-07-28/checkpoints/checkpoint.pth", map_location='cpu')  # Modify the path to your checkpoint
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
all_latents = []
import pandas as pd
df = pd.read_csv("/Data/summer24/DreamDiffusion/datasets/eegdata/train/subject.csv",index_col=0)

with torch.no_grad():
    all_latents = []
    for iter, data_dict in enumerate(valid_dataloader_eeg):
        sample = data_dict['eeg']
        latent, _, _ = model.forward_encoder(sample, mask_ratio=config.mask_ratio)
        latent = latent[:, 1:, :]  # Remove the cls token
        batch_size = latent.size(0)
        latent = latent.view(batch_size, -1) 
        all_latents.append(latent.cpu())

all_latents = np.concatenate(all_latents, axis=0)
all_labels = df.values


tsne = TSNE(n_components=2,perplexity=40)
latent_tsne = tsne.fit_transform(all_latents)

plt.figure(figsize=(10, 8))
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('2D t-SNE Visualization of Latent Representations')
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1],c=all_labels, alpha=0.5)
plt.colorbar()
plt.show()
plt.savefig('/Data/summer24/DreamDiffusion/result/p40_11-07_checkpoint_tsne_train_subject.png')