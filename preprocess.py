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
import shutil

with open('./data/trainLabels.csv', 'r') as f:
    lines = f.readlines()[1:]
    tokens = [i.rstrip().split(',') for i in lines]
    idx_label = dict((int(idx), label) for idx, label in tokens)
labels = set(idx_label.values())

num_train = len(os.listdir('./data/train/'))

num_train_tuning = int(num_train * (1 - 0.1))

num_train_tuning_per_label = num_train_tuning // len(labels)

label_count = dict()


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


for train_file in os.listdir('./data/train/'):
    idx = int(train_file.split('.')[0])
    label = idx_label[idx]
    mkdir_if_not_exist(['./data', 'train_valid', label])
    shutil.copy(os.path.join('./data/train/', train_file),
                os.path.join('./data/train_valid', label))
    if label not in label_count or label_count[label] < num_train_tuning_per_label:
        mkdir_if_not_exist(['./data/train_data', label])
        shutil.copy(os.path.join('./data/train', train_file),
                    os.path.join('./data/train_data', label))
        label_count[label] = label_count.get(label, 0) + 1
    else:
        mkdir_if_not_exist(['./data/valid_data', label])
        shutil.copy(os.path.join('./data/train/', train_file),
                    os.path.join('./data/valid_data', label))
