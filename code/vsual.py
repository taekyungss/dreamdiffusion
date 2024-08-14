from sc_mbm.mae_for_eeg_2 import MAEforEEG
from config import Config_MBM_EEG
from stageA1_eeg_pretrain import fmri_transform
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from sklearn.manifold import TSNE
import numpy as np
from eegtoimage.dataset import EEGDataset_subject
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import torch.nn as nn

config = Config_MBM_EEG()
local_rank = config.local_rank
device = torch.device(f'cuda:{local_rank}')

# dataset 경로 설정 및 데이터 로드
dataset = EEGDataset_subject(eeg_signals_path='/Data/summer24/DreamDiffusion/datasets/eegdata_subject2/combined_data_2.pth', mode = "dataset")
dataloader_eeg = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

# 모델 초기화 및 체크포인트 로드
model = MAEforEEG(time_len=dataset.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,
                  decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                  num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                  focus_range=config.focus_range, focus_rate=config.focus_rate, 
                  img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss, num_classes=config.num_classes)   

checkpoint = torch.load("/Data/summer24/DreamDiffusion/DreamDiffuion/results/eeg_pretrain/30-07-2024-01-30-34/checkpoints/checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# 레이블 데이터 로드 및 인코딩
df = pd.read_csv("/Data/summer24/DreamDiffusion/datasets/eegdata_subject2/label.csv")
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'].values)

with torch.no_grad():
    all_latents = []
    for iter, data_dict in enumerate(dataloader_eeg):
        sample = data_dict['eeg']
        latent, _, _ = model.forward_encoder(sample, mask_ratio=config.mask_ratio)
        latent = latent[:, 1:, :]  # cls 토큰 제거
        batch_size = latent.size(0)
        latent = latent.view(batch_size, -1) 
        all_latents.append(latent.cpu())

# 잠재 표현 결합
all_latents = np.concatenate(all_latents, axis=0)
all_labels = df.values

# t-SNE 적용
tsne = TSNE(n_components=2)
latent_tsne = tsne.fit_transform(all_latents)

# 시각화
plt.figure(figsize=(10, 8))
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('2D t-SNE Visualization of Latent Representations')

# 데이터 포인트 시각화
highlight_indices = np.arange(0, 1330)
non_highlight_indices = np.arange(1330, all_latents.shape[0])

plt.scatter(latent_tsne[highlight_indices, 0], latent_tsne[highlight_indices, 1], 
            edgecolors='black', linewidths=1.5, c='red', alpha=0.7, label='0-1329 (Red)')
plt.scatter(latent_tsne[non_highlight_indices, 0], latent_tsne[non_highlight_indices, 1], 
            edgecolors='black', linewidths=1.5, c='blue', alpha=0.7, label='1330+ (Blue)')

plt.legend()
plt.colorbar()
plt.show()
plt.savefig('/Data/summer24/DreamDiffusion/tsne_result/29-07_tsne_img_label_total_dataset_puddle_highlighted.png')


