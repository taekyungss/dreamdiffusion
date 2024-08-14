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

        self.data = []
        for item in chosen_data:
            image_name = item["image"]
            image_path = os.path.join(self.dataset_dir, "imageNet", image_name.split('_')[0], image_name)
            if os.path.exists(image_path):
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

            image = image_raw
            image_raw = self.processor(images=image_raw, return_tensors="pt")
            image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

            if label.mode == 'L':
                label = label.convert('RGB')
            if self.transform:
                label = self.transform(label)
            else:
                label = path
        else:
            label = self.labels.index(self.data[index]["label"])
            image = None

        if self.use_frequency_feat:
            feat = self.frequency_feat[index]
        else:
            eeg_data = self.data[index]["eeg_data"].float()
            feat = eeg_data[:, 40:440]
    
        return {'eeg': feat, 'label': label, 'image': self.transform(image), 'image_raw': image_raw}


    def __len__(self):
        return len(self.data)

    def add_frequency_feat(self, feat):
        if len(feat) == len(self.data):
            self.frequency_feat = torch.from_numpy(feat).float()
        else:
            raise ValueError("Frequency features must have same length")


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