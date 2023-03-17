import os, io, h5py, numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
from PIL import Image

import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F


class Flowers102(Dataset):

    def __init__(self, datasetFile='./data/flowers.hdf5', transform=None, split='train'):
        assert split in ['train', 'valid', 'test'], f'split should be \'train\'/\'val\'/\'test\'. Got {split} instead'
        self.datasetFile = datasetFile
        self.split = split
        self.transform = transform
        self.dataset = h5py.File(self.datasetFile, mode='r')
        self.dataset_keys = [str(k) for k in self.dataset[self.split].keys()][:64]    # FIXME: Change back later
        self.h5py2int = lambda x: int(np.array(x))
        self.transform = transform

    def __len__(self): return len(self.dataset_keys)

    def __getitem__(self, idx):
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]


        txt = np.array(example['txt']).astype(str)


        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        right_image = self.validate_image(right_image)
        wrong_image = self.validate_image(wrong_image)

        sample = {
            'right_images': torch.FloatTensor(right_image),
            'right_embed': torch.FloatTensor(right_embed),
            'wrong_images': torch.FloatTensor(wrong_image),
            'inter_embed': torch.FloatTensor(inter_embed),
            'txt': str(txt)
        }
        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']


    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)
