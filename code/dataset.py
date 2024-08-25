from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
from sklearn.preprocessing import LabelEncoder
import torch
from pathlib import Path
import torchvision.transforms as transforms
from transformers import AutoProcessor
from scipy.interpolate import interp1d
from PIL import Image

def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]
    

class eeg_pretrain_dataset(Dataset):
    def __init__(self, path, roi='VC', patch_size=16, transform=None, aug_times=2, 
                 num_sub_limit=None, include_kam=False, include_hcp=True):
        super(eeg_pretrain_dataset, self).__init__()
        
        self.transform = transform if transform is not None else lambda x: x
        self.input_paths = [str(f) for f in sorted(Path(path).rglob('*')) if f.suffix == '.npy' and os.path.isfile(f)]
        
        if not self.input_paths:
            raise ValueError('No data found')

        self.data_len = 1024
        self.data_chan = 128


    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        data_path = self.input_paths[index]

        data = np.load(data_path)

        if data.shape[-1] > self.data_len:
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)

            data = data[:, idx: idx+self.data_len]
        else:
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)
        ret = np.zeros((self.data_chan, self.data_len))
        if (self.data_chan > data.shape[-2]):
            for i in range((self.data_chan//data.shape[-2])):

                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] != 0:

                ret[ -(self.data_chan%data.shape[-2]):, :] = data[: (self.data_chan%data.shape[-2]), :]
        elif(self.data_chan < data.shape[-2]):
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]
        # print(ret.shape)
        elif(self.data_chan == data.shape[-2]):
            ret = data
        ret = ret/10 # reduce an order
        # torch.tensor()
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret }



class EEGDataset_r(Dataset):

    # Constructor
    def __init__(self, image_transform=identity):

        self.imagesource = '/Data/summer24/DreamDiffusion/datasets/imageNet_images'
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 1024
        # # Compute size
        self.size = 100

    # Get size
    def __len__(self):
        return 100

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = torch.randn(128,1024)

        # print(image.shape)
        label = torch.tensor(0).long()
        image = torch.randn(3,675,675)
        image_raw = image

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}


class EEGDataset_s(Dataset):

    # Constructor
    def __init__(self, image_transform, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        self.eeg = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagesource = '/Data/summer24/DreamDiffusion/datasets/imageNet_images'
        self.image_transform = image_transform
        self.num_voxels = 1024
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()

        # Get label
        image_name = self.images[self.data[i]["image"]]
        # image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        return image_name



class EEGDataset(Dataset):
    
    # Constructor
    def __init__(self, eeg_signals_path, image_transform=identity, subject = 0):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        # print(loaded)
        if subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==subject]
        else:
            self.data = loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = '/Data/summer24/DreamDiffusion/datasets/imageNet_images'
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        # Compute size
        self.size = len(self.data)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        # print(self.data[i])
        eeg = self.data[i]["eeg"].float().t()

        eeg = eeg[20:460,:]
        ##### 2023 2 13 add preprocess and transpose
        eeg = np.array(eeg.transpose(0,1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        ##### 2023 2 13 add preprocess
        label = torch.tensor(self.data[i]["label"]).long()

        # Get label
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        # print(image_path)
        image_raw = Image.open(image_path).convert('RGB') 
        
        image = np.array(image_raw) / 255.0
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)


        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}
        # Return
        # return eeg, label


# train -> single / train_data -> multi
class EEGDataset_subject(Dataset):
    def __init__(self, eeg_signals_path, mode="train"):
        loaded = torch.load(eeg_signals_path)
        if mode == "train":
            self.dataset = loaded['train']
            # self.dataset = loaded['train_data']
        elif mode == "val":
            self.dataset = loaded['val']
            # self.dataset = loaded['val_data']
        else:
            self.dataset = loaded

        self.data_len = 512
        self.size = len(self.dataset)

        labels = [self.dataset[i]["label"] for i in range(self.size)]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.dataset[i]["eeg"].float().t()
        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        label = torch.tensor(self.encoded_labels[i]).long()

        return {'eeg': eeg, 'label': label}


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def create_EEG_dataset(eeg_signals_path='../DreamDiffusion/datasets/eeg_5_95_std.pth', 
            splits_path = '../DreamDiffusion/datasets/block_splits_by_image_single.pth',
            # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth',
            image_transform=identity, subject = 0):
    # if subject == 0:
        # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth'
    if isinstance(image_transform, list):
        dataset_train = EEGDataset(eeg_signals_path, image_transform[0], subject )
        dataset_test = EEGDataset(eeg_signals_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset(eeg_signals_path, image_transform, subject)
        dataset_test = EEGDataset(eeg_signals_path, image_transform, subject)
    split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train', subject= subject)
    split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test', subject = subject)
    return (split_train, split_test)




def create_EEG_dataset_r(eeg_signals_path='../DreamDiffusion/datasets/eeg_5_95_std.pth', 
            # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_single.pth',
            splits_path = '../DreamDiffusion/datasets/block_splits_by_image_all.pth',
            image_transform=identity):
    if isinstance(image_transform, list):
        dataset_train = EEGDataset_r(eeg_signals_path, image_transform[0])
        dataset_test = EEGDataset_r(eeg_signals_path, image_transform[1])
    else:
        dataset_train = EEGDataset_r(eeg_signals_path, image_transform)
        dataset_test = EEGDataset_r(eeg_signals_path, image_transform)
    # split_train = Splitter(dataset_train, split_path = splits_path, split_num = 0, split_name = 'train')
    # split_test = Splitter(dataset_test, split_path = splits_path, split_num = 0, split_name = 'test')
    return (dataset_train,dataset_test)
