# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import time

import numpy as np
import torch

from utils.meter import AverageValueMeter
from utils.serialization import save_checkpoint


class Solver(object):
    def __init__(self, net, use_gpu):
        self.net = net
        self.use_gpu = use_gpu
        self.loss = AverageValueMeter()
        self.acc = AverageValueMeter()

    def fit(self, train_data, test_data, optimizer, criterion, num_epochs=100, print_interval=100,
            eval_step=50, save_step=10, save_dir='checkpoints'):
        best_test_acc = -np.inf
        for epoch in range(num_epochs):
            self.loss.reset()
            self.acc.reset()
            self.net.train()

            tic = time.time()
            btic = time.time()
            for i, data in enumerate(train_data):
                imgs, labels = data
                if self.use_gpu:
                    labels = labels.cuda()
                scores = self.net(imgs)
                loss = criterion(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss.add(loss.item())
                acc = (scores.max(1)[1] == labels.long()).float().mean()
                self.acc.add(acc.item())

                if print_interval and not (i + 1) % print_interval:
                    loss_mean = self.loss.value()[0]
                    acc_mean = self.acc.value()[0]
                    logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tloss=%f\t'
                                 'acc=%f' % (
                                     epoch, i + 1, train_data.batch_size * print_interval / (time.time() - btic),
                                     loss_mean, acc_mean))
                    btic = time.time()

            loss_mean = self.loss.value()[0]
            acc_mean = self.acc.value()[0]
            throughput = int(train_data.batch_size * len(train_data) / (time.time() - tic))

            logging.info('[Epoch %d] training: loss=%f\tacc=%f' % (
                epoch, loss_mean, acc_mean))
            logging.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))

            is_best = False
            if test_data is not None and eval_step and not (epoch + 1) % eval_step:
                test_acc = self.test_func(test_data)
                is_best = test_acc > best_test_acc
                if is_best:
                    best_test_acc = test_acc
            state_dict = self.net.module.state_dict()
            if not (epoch + 1) % save_step:
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': epoch + 1,
                }, is_best=is_best, save_dir=save_dir,
                    filename='model' + '.pth.tar')

    def test_func(self, test_data) -> float:
        num_correct = 0
        num_imgs = 0
        self.net.eval()
        for data in test_data:
            imgs, labels = data
            if self.use_gpu:
                labels = labels.cuda()
            with torch.no_grad():
                scores = self.net(imgs)
            num_correct += (scores.max(1)[1] == labels).float().sum().item()
            num_imgs += imgs.shape[0]
        return num_correct / num_imgs
