import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from eegtoimage.dataset import EEGDataset_subject
from sc_mbm.mae_for_eeg_2 import MAEforEEG
from config import Config_MBM_EEG
from stageA1_eeg_pretrain import fmri_transform
import matplotlib.pyplot as plt
import seaborn as sns


config = Config_MBM_EEG()
local_rank = config.local_rank
device = torch.device(f'cuda:{local_rank}')

# dataset 및 dataloader 초기화
dataset = EEGDataset_subject(eeg_signals_path='/Data/summer24/DreamDiffusion/datasets/eegdata_subject2/combined_data.pth', mode="dataset")
dataloader_eeg = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

# 모델 초기화
model = MAEforEEG(time_len=dataset.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,
                  decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                  num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                  focus_range=config.focus_range, focus_rate=config.focus_rate, 
                  img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss, num_classes=config.num_classes)   

checkpoint = torch.load("/Data/summer24/DreamDiffusion/DreamDiffuion/results/eeg_pretrain/30-07-2024-01-30-34/checkpoints/checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# 레이블 로드 및 변환
df = pd.read_csv("/Data/summer24/DreamDiffusion/datasets/eegdata_subject2/labels.csv")
all_labels = df['label'].values

with torch.no_grad():
    all_latents = []
    for iter, data_dict in enumerate(dataloader_eeg):
        sample = data_dict['eeg']
        latent, _, _ = model.forward_encoder(sample, mask_ratio=config.mask_ratio)
        latent = latent[:, 1:, :]  # Remove the cls token
        batch_size = latent.size(0)
        latent = latent.view(batch_size, -1) 
        all_latents.append(latent.cpu().numpy())

# Concatenate all latent representations
all_latents = np.concatenate(all_latents, axis=0)

# t-SNE 시각화
tsne = TSNE(n_components=2)
latent_tsne = tsne.fit_transform(all_latents)


# 색상 팔레트 정의
palette = sns.color_palette("hsv", len(set(all_labels)))

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=all_labels, cmap=plt.get_cmap('tab10', len(set(all_labels))), s=10, alpha=0.7)

highlight_indices = np.arange(1330, len(all_latents))
highlight_points = latent_tsne[highlight_indices]
highlight_labels = all_labels[highlight_indices]

# 윤곽선 스타일 적용 (작은 포인트, 얇은 윤곽선)
plt.scatter(highlight_points[:, 0], highlight_points[:, 1], edgecolor='r', facecolor='none', s=15, alpha=0.5, linewidth=0.5, label='Highlighted Points')


# 레전드 및 제목 설정
plt.colorbar(scatter, ticks=range(len(set(all_labels))), label='Label')
plt.title('t-SNE Visualization of Latent Representations')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('/Data/summer24/DreamDiffusion/tsne_result/30-07_tsne_data.png')
