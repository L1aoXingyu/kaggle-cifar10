# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def get_data_provider(batch_size: int) -> tuple:
    train_aug = T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.64, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])

    test_aug = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])

    train_set = ImageFolder('data/train_data', train_aug)
    valid_set = ImageFolder('data/valid_data', test_aug)
    train_valid_set = ImageFolder('data/train_valid', train_aug)

    train_loader = DataLoader(train_set, batch_size, True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size, False, num_workers=8, pin_memory=True)
    train_valid_loader = DataLoader(train_valid_set, batch_size, True, num_workers=8, pin_memory=True)
    return train_loader, valid_loader, train_valid_loader


class TestSet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.img_list = os.listdir(root)
        self.transform = transform

    def __getitem__(self, item):
        fname = self.img_list[item]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, int(fname.split('.')[0])

    def __len__(self):
        return len(self.img_list)


def get_test_provider(batch_size):
    train_set = ImageFolder('data/train_data')
    id_to_class = {j: i for i, j in train_set.class_to_idx.items()}
    test_aug = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.491, 0.482, 0.446], std=[0.202, 0.199, 0.201])
    ])
    test_set = TestSet('data/test', test_aug)
    test_loader = DataLoader(test_set, batch_size, num_workers=8, pin_memory=True)
    return test_loader, id_to_class
