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
import sys

import torch
from torch import nn
from torch.backends import cudnn

import network
from core.loader import get_data_provider
from core.solver import Solver

FORMAT = '[%(levelname)s]: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    stream=sys.stdout
)


def train(args):
    train_data, valid_data, train_valid_data = get_data_provider(args.bs)

    net = network.ResNet18(num_classes=10)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    ce_loss = nn.CrossEntropyLoss()

    net = nn.DataParallel(net)
    if args.use_gpu:
        net = net.cuda()
    mod = Solver(net, args.use_gpu)
    mod.fit(train_data=train_data, test_data=valid_data, optimizer=optimizer, criterion=ce_loss,
            num_epochs=args.epochs, print_interval=args.print_interval, eval_step=args.eval_step,
            save_step=args.save_step, save_dir=args.save_dir)


def main():
    parser = argparse.ArgumentParser(description='cifar10 model training')
    parser.add_argument('--bs', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='training learning rate')
    parser.add_argument('--wd', type=float, default=3e-4, help='training weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum training')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--print-interval', type=int, default=300, help='how many iterations to print')
    parser.add_argument('--eval-step', type=int, default=20, help='how many epochs to evaluate')
    parser.add_argument('--save-step', type=int, default=20, help='how many epochs to save model')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='save model directory')
    parser.add_argument('--use-gpu', type=bool, default=True, help='decide if use gpu training')

    args = parser.parse_args()
    cudnn.benchmark = True
    train(args)


if __name__ == '__main__':
    main()
