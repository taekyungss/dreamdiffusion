
import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
import torchvision.transforms as transforms



class EEGImageNetDataset(Dataset):
    def __init__(self, args, transform=None):
        self.dataset_dir = args.dataset_dir
        self.transform = transform
        loaded = torch.load(os.path.join(args.dataset_dir, "EEG-ImageNet.pth"))
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        if args.subject != -1:
            chosen_data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                           loaded['dataset'][i]['subject'] == args.subject]
        else:
            chosen_data = loaded['dataset']
        if args.granularity == 'coarse':
            self.data = [i for i in chosen_data if i['granularity'] == 'coarse']
        elif args.granularity == 'all':
            self.data = chosen_data
        else:
            fine_num = int(args.granularity[-1])
            fine_category_range = np.arange(8 * fine_num, 8 * fine_num + 8)
            self.data = [i for i in chosen_data if
                         i['granularity'] == 'fine' and self.labels.index(i['label']) in fine_category_range]

        # 실제로 이미지가 존재하는 데이터만 남김
        self.data = []
        for item in chosen_data:
            image_name = item["image"]
            image_path = os.path.join(self.dataset_dir, "imageNet", image_name.split('_')[0], image_name)
            if os.path.exists(image_path):  # 이미지 파일이 실제로 존재하는지 확인
                self.data.append(item)

        self.use_frequency_feat = False
        self.frequency_feat = None
        self.use_image_label = True
        self.imagenet = os.path.join(args.dataset_dir, "imageNet")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __getitem__(self, index):
        if self.use_image_label:
            path = self.data[index]["image"]
            label_path = os.path.join(self.dataset_dir, "imageNet", path.split('_')[0], path)
            label = None
            try:
                label = Image.open(label_path)
            except FileNotFoundError:
                return self.__getitem__(index + 1)
            
            image_name = self.data[index]["image"]
            image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name)
            image_raw = Image.open(image_path).convert('RGB')

            image_raw = self.processor(images=image_raw, return_tensors="pt")
            image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

            if label.mode == 'L':
                label = label.convert('RGB')

            if self.transform is not None:
                image = self.transform(image_raw['pixel_values'])

            else:
                image = image_raw['pixel_values']

            label = self.labels.index(self.data[index]["label"])
        else:
            return None

        if self.use_frequency_feat:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]
    
        return {'eeg': feat, 'label': label, 'image': image, 'image_raw': image_raw}


    def __len__(self):
        return len(self.data)


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

class hcp_dataset(Dataset):
    def __init__(self, path='../data/HCP/npz', roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True):
        super(hcp_dataset, self).__init__()
        data = []
        images = []
        
        if include_hcp:
            for c, sub in enumerate(os.listdir(path)):
                if os.path.isfile(os.path.join(path,sub,'HCP_visual_voxel.npz')) == False:
                    continue 
                if num_sub_limit is not None and c > num_sub_limit:
                    break
                npz = dict(np.load(os.path.join(path,sub,'HCP_visual_voxel.npz')))
                voxels = np.concatenate([npz['V1'],npz['V2'],npz['V3'],npz['V4']], axis=-1) if roi == 'VC' else npz[roi] # 1200, num_voxels
                voxels = process_voxel_ts(voxels, patch_size) # num_samples, num_voxels_padded
                data.append(voxels)
                
            data = augmentation(np.concatenate(data, axis=0), aug_times) # num_samples, num_voxels_padded
            data = np.expand_dims(data, axis=1) # num_samples, 1, num_voxels_padded
            images += [None] * len(data)

        if include_kam:
            kam_path = os.path.join(str(Path(path).parent.parent), 'Kamitani', 'npz')
            k = Kamitani_pretrain_dataset(kam_path, roi, patch_size, transform, aug_times)
            if len(data) != 0:
                padding_len = max([data.shape[-1],  k.data.shape[-1]])
                data = pad_to_length(data, padding_len)
                data_k = pad_to_length(k.data, padding_len)
                data = np.concatenate([data, data_k], axis=0)
            else:
                data = k.data
            images += k.images

        assert len(data) != 0, 'No data found'
        
        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.data = data
        self.transform = transform
        self.images = images
        self.images_transform = transforms.Compose([
                                            img_norm,
                                            transforms.Resize((112, 112)), 
                                            channel_first
                                        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.images[index]
        images_transform = self.images_transform if img is not None else identity
        img = img if img is not None else torch.zeros(3, 112, 112)

        return {'fmri': self.transform(self.data[index]),
                'image': images_transform(img)}
       
class Kamitani_pretrain_dataset(Dataset):
    def __init__(self, path='../data/Kamitani/npz', roi='VC', patch_size=16, transform=identity, aug_times=2):
        super(Kamitani_pretrain_dataset, self).__init__()
        k1, k2 = create_Kamitani_dataset(path, roi, patch_size, transform, include_nonavg_test=True)
        # data = np.concatenate([k1.fmri, k2.fmri], axis=0)
        # self.images = [img for img in k1.image] + [None] * len(k2.fmri)

        data = k1.fmri
        self.images = [(img*255.0).astype(np.uint8) for img in k1.image]

        # data = augmentation(data, aug_times)
        self.data = np.expand_dims(data, axis=1)
        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.transform(self.data[index])

def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

def create_Kamitani_dataset(path='../data/Kamitani/npz',  roi='VC', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], 
            test_category=None, include_nonavg_test=False):
    img_npz = dict(np.load(os.path.join(path, 'images_256.npz')))
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)

    with open(os.path.join(path, 'imagenet_training_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_training_filename = [row for row in csvreader]

    with open(os.path.join(path, 'imagenet_testing_label.csv'), 'r') as f:
        csvreader = csv.reader(f)
        img_testing_filename = [row for row in csvreader]

    train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
    test_img_label, _ = get_img_label(img_class_index, img_testing_filename, naive_label_set)

    test_img = [] # img_npz['test_images']
    train_img = [] # img_npz['train_images']
    train_fmri = []
    test_fmri = []
    train_img_label_all = []
    test_img_label_all = []
    for sub in subjects:
        npz = dict(np.load(os.path.join(path, f'{sub}.npz')))
        test_img.append(img_npz['test_images'])
        train_img.append(img_npz['train_images'][npz['arr_3']])
        train_lb = [train_img_label[i] for i in npz['arr_3']]
        test_lb = test_img_label
        
        roi_mask = npz[roi]
        tr = npz['arr_0'][..., roi_mask] # train
        tt = npz['arr_2'][..., roi_mask] 
        if include_nonavg_test:
            tt = np.concatenate([tt, npz['arr_1'][..., roi_mask]], axis=0)

        # train_fmri.append(tr[..., :tr.shape[-1] - tr.shape[-1] % patch_size])
        # test_fmri.append(tt[..., :tt.shape[-1] - tt.shape[-1] % patch_size])
        tr = normalize(pad_to_patch_size(tr, patch_size))
        tt = normalize(pad_to_patch_size(tt, patch_size), np.mean(tr), np.std(tr))
        train_fmri.append(tr)
        test_fmri.append(tt)
        if test_category is not None:
            train_img_, train_fmri_, test_img_, test_fmri_, train_lb, test_lb = reorganize_train_test(train_img[-1], train_fmri[-1], 
                                                            test_img[-1], test_fmri[-1], train_lb, test_lb,
                                                            test_category, npz['arr_3'])
            train_img[-1] = train_img_
            train_fmri[-1] = train_fmri_
            test_img[-1] = test_img_
            test_fmri[-1] = test_fmri_
        
        train_img_label_all += train_lb
        test_img_label_all += test_lb

    len_max = max([i.shape[-1] for i in test_fmri])
    test_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in test_fmri]
    train_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in train_fmri]

    # len_min = min([i.shape[-1] for i in test_fmri])
    # test_fmri = [i[:,:len_min] for i in test_fmri]
    # train_fmri = [i[:,:len_min] for i in train_fmri]


    test_fmri = np.concatenate(test_fmri, axis=0)
    train_fmri = np.concatenate(train_fmri, axis=0)
    test_img = np.concatenate(test_img, axis=0)
    train_img = np.concatenate(train_img, axis=0)
    num_voxels = train_fmri.shape[-1]

    # test_img = rearrange(test_img, 'n h w c -> n c h w')
    # train_img = rearrange(train_img, 'n h w c -> n c h w')

    if isinstance(image_transform, list):
        return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform[0], num_voxels, len(npz['arr_0'])), 
                Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform[1], num_voxels, len(npz['arr_2'])))
    else:
        return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform, num_voxels, len(npz['arr_0'])), 
                Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_2'])))

def reorganize_train_test(train_img, train_fmri, test_img, test_fmri, train_lb, test_lb, 
                    test_category, train_index_lookup):
    test_img_ = []
    test_fmri_ = []
    test_lb_ = []
    train_idx_list = []
    num_per_category = 8
    for c in test_category:
        c_idx = c * num_per_category + np.random.choice(num_per_category, 1)[0]
        train_idx = train_index_lookup[c_idx]
        test_img_.append(train_img[train_idx])
        test_fmri_.append(train_fmri[train_idx])
        test_lb_.append(train_lb[train_idx])
        train_idx_list.append(train_idx)
    
    train_img_ = np.stack([img for i, img in enumerate(train_img) if i not in train_idx_list])
    train_fmri_ = np.stack([fmri for i, fmri in enumerate(train_fmri) if i not in train_idx_list])
    train_lb_ = [lb for i, lb in enumerate(train_lb) if i not in train_idx_list] + test_lb

    train_img_ = np.concatenate([train_img_, test_img], axis=0)
    train_fmri_ = np.concatenate([train_fmri_, test_fmri], axis=0)

    test_img_ = np.stack(test_img_)
    test_fmri_ = np.stack(test_fmri_)
    return train_img_, train_fmri_, test_img_, test_fmri_, train_lb_, test_lb_

class Kamitani_dataset(Dataset):
    def __init__(self, fmri, image, img_label, fmri_transform=identity, image_transform=identity, num_voxels=0, num_per_sub=50):
        super(Kamitani_dataset, self).__init__()
        self.fmri = fmri
        self.image = image
        if len(self.image) != len(self.fmri):
            self.image = np.repeat(self.image, 35, axis=0)
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.num_per_sub = num_per_sub
        self.img_class = [i[0] for i in img_label]
        self.img_class_name = [i[1] for i in img_label]
        self.naive_label = [i[2] for i in img_label]
        self.return_image_class_info = False

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        if index >= len(self.image):
            img = np.zeros_like(self.image[0])
        else:
            img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) # (1, num_voxels)
        if self.return_image_class_info:
            img_class = self.img_class[index]
            img_class_name = self.img_class_name[index]
            naive_label = torch.tensor(self.naive_label[index])
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img),
                    'image_class': img_class, 'image_class_name': img_class_name, 'naive_label':naive_label}
        else:
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}

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
    
def create_BOLD5000_dataset(path='../data/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []
    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))
      
        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)

        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform[0], num_voxels), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform[1], num_voxels))
    else:
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform, num_voxels), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform, num_voxels))

class BOLD5000_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity, num_voxels=0):
        self.fmri = fmri
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
    
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) 
        return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}
    
    def switch_sub_view(self, sub, subs):
        # Not implemented
        pass


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

        self.imagesource = 'mind-vis/datasets/imageNet_images'
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
        self.imagesource = 'mind-vis/datasets/imageNet_images'
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
        self.imagenet = 'mind-vis/datasets/imageNet_images'
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
        label = torch.tensor(self.data[i]["label"]).long()

        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[0], image_name+'.JPEG')
        image_raw = Image.open(image_path).convert('RGB') 
        
        image = np.array(image_raw) / 255.0
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)


        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}


class EEGDataset_subject(Dataset):
    def __init__(self, eeg_signals_path,mode = "train"):

        loaded = torch.load(eeg_signals_path)
        if mode == "train":
            self.dataset = loaded['train']
        else:
            self.dataset = loaded['val']

        self.data_len = 512
        self.size = len(self.dataset)

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

        label = torch.tensor(self.dataset[i]["label"]).long()

        return {'eeg': eeg, 'label': label}



# class Args:
#     dataset_dir = '/Data/summer24/data'
#     subject = -1
#     granularity = 'all'

# args = Args()
# if __name__=="__main__":

#     transform = transforms.Compose([
#         transforms.Resize(256), 
#         transforms.CenterCrop(224),
#         transforms.ToTensor(), 
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     dataset = EEGImageNetDataset(args, transform) 
#     print(len(dataset))
