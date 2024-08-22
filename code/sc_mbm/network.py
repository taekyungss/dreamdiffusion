import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config_Generative_Model
np.random.seed(45)
torch.manual_seed(45)


class EEGFeatNet(nn.Module):
    def __init__(self, n_classes=40, in_channels=128, n_features=128, projection_dim=128, num_layers=1):
        super(EEGFeatNet, self).__init__()
        self.hidden_size= n_features
        self.num_layers = num_layers
        # self.embedding  = nn.Embedding(num_embeddings=in_channels, embedding_dim=n_features)
        self.encoder    = nn.LSTM(input_size=in_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc         = nn.Linear(in_features=n_features, out_features=projection_dim, bias=False)
        

    def forward(self, x):
        config = Config_Generative_Model()
        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device) 
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.device)
        output, (h_n, c_n) = self.encoder( x, (h_n, c_n) )
        feat = output
        x = self.fc(feat)
        x = F.normalize(x, dim=-1)
        # # print(x.shape, feat.shape)
        return x
        

    def load_checkpoint(self, state_dict):
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 

# if __name__ == '__main__':

#     eeg   = torch.randn((512, 440, 128)).to(config.device)
#     model = EEGFeatNet(n_features=config.feat_dim, projection_dim=config.projection_dim).to(config.device)
#     proj, feat = model(eeg)
#     print(proj.shape, feat.shape)
#     # feat  = model(eeg)
#     # print(feat.shape)