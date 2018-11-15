# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys

import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn

import network
from core.loader import get_test_provider

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def submission(args):
    test_loader, id_to_class = get_test_provider(args.bs)
    net = network.ResNet18(num_classes=10)
    net.load_state_dict(torch.load(args.model_path)['state_dict'])
    net = nn.DataParallel(net)
    net.eval()
    if args.use_gpu:
        net = net.cuda()

    pred_labels = list()
    indices = list()
    for data, fname in test_loader:
        if args.use_gpu:
            data = data.cuda()
        with torch.no_grad():
            scores = net(data)
        labels = scores.max(1)[1].cpu().numpy()
        pred_labels.extend(labels)
        indices.extend(fname.numpy())
    df = pd.DataFrame({'id': indices, 'label': pred_labels})
    df['label'] = df['label'].apply(lambda x: id_to_class[x])
    df.to_csv('submission.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='cifar10 model testing')
    parser.add_argument('--model_path', type=str, default='checkpoints/model_best.pth.tar',
                        help='training batch size')
    parser.add_argument('--bs', type=int, default=128, help='testing batch size')
    parser.add_argument('--use_gpu', action='store_true', help='decide if use gpu training')

    args = parser.parse_args()
    cudnn.benchmark = True
    submission(args)


if __name__ == '__main__':
    main()
