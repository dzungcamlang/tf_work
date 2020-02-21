#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 16:35 
# @Author  : Ming Liu
# @License : Ali Licence
# @Site    :
# @File    : save.py 
# @Software: PyCharm Community Edition

import os
import time

import tensorflow as tf

from util.loss import *
from util.data import *

class Solver(object):

    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Low frame rate feature
        self.LFR_m = args.LFR_m
        self.LFR_n = args.LFR_n

        # Training config
        self.epochs = args.epochs
        self.label_smoothing = args.label_smoothing
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):

        checkpoint_path = self.model_path
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=30)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

                file_path = ckpt_manager.save()
                print("Find better validated model, saving to %s" % file_path)


    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
                padded_input, input_lengths, padded_target = data
                pred, gold = self.model(padded_input, input_lengths, padded_target)
                loss, n_correct = cal_performance(pred, gold, smoothing=self.label_smoothing)

                gradients = tf.GradientTape().gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # if not cross_valid:
                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     self.optimizer.step()
                loss_mean = tf.keras.metrics.Mean(name='loss_mean')
                total_loss = loss_mean(loss)
                non_pad_mask = gold.ne(IGNORE_ID)
                n_word = non_pad_mask.sum().item()

                if i % self.print_freq == 0:
                    print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                          'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1),
                        loss.item(), 1000 * (time.time() - start) / (i + 1)),
                        flush=True)

        return total_loss
