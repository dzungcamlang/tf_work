#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/20 10:01 
# @Author  : Ming Liu
# @License : Ali Licence
# @Site    :
# @File    : e2e_transformer_train.py 
# @Software: PyCharm Community Edition

import argparse
import tensorflow as tf

from util.data import *
from util.save import *
from util.loss import *
from util.optimizer import *
from model.transformer.encoder import Encoder
from model.transformer.decoder import Decoder
from model.transformer.transformer import Transformer

parser = argparse.ArgumentParser(" End-to-End Automatic Speech Recognition Training. ")

# Data Config
parser.add_argument('--train-json', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid-json', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='final.pth.tar',
                    help='Location to save best validation model')

# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')

# Network Config
# encoder
parser.add_argument('--n_layers_enc', default=6, type=int,
                    help='Number of encoder stacks')
parser.add_argument('--n_heads', default=8, type=int,
                    help='Number of Multi Head Attention (MHA)')
parser.add_argument('--d_model', default=512, type=int,
                    help='Dimension of model')
parser.add_argument('--d_dff', default=2048, type=int,
                    help='Dimension of dff')
parser.add_argument('--dropout_rate', default=0.1, type=float,
                    help='Dropout rate')
parser.add_argument('--pe_maxlen', default=10000, type=int,
                    help='Positional Encoding max len')
# decoder
parser.add_argument('--d_word_vec', default=512, type=int,
                    help='Dim of decoder embedding')
parser.add_argument('--n_layers_dec', default=6, type=int,
                    help='Number of decoder stacks')
parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                    help='share decoder embedding with decoder projection')
parser.add_argument('--label_smoothing', default=0.1, type=float,
                    help='label smoothing')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--shuffle', default=0, type=int,
                    help='Reshuffle the data at every epoch')
parser.add_argument('--batch-size', default=64, type=int,
                    help='Batch size')
parser.add_argument('--batch_frames', default=0, type=int,
                    help='Batch frames. If this is not 0, batch size will make no sense')
parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
parser.add_argument('--warmup_steps', default=4000, type=int,
                    help='Warmup steps')

def main(args):

    tr_dataset = AudioDataset(args.train_json,
                              args.batch_size,
                              args.maxlen_in,
                              args.maxlen_out,
                              batch_frames=args.batch_frames)
    cv_dataset = AudioDataset(args.valid_json,
                              args.batch_size,
                              args.maxlen_in,
                              args.maxlen_out,
                              batch_frames=args.batch_frames)
    tr_loader = AudioDataLoader(tr_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                shuffle=args.shuffle,
                                LFR_m=args.LFR_m,
                                LFR_n=args.LFR_n)
    cv_loader = AudioDataLoader(cv_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m,
                                LFR_n=args.LFR_n)
    # load dictionary and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.dict)
    vocab_size = len(char_list)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    encoder = Encoder(args.n_layers_enc,
                      args.n_heads,
                      args.d_model,
                      args.d_dff,
                      dropout_rate=args.dropout_rate,
                      pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id,
                      eos_id,
                      vocab_size,
                      args.d_word_vec,
                      args.n_layers_dec,
                      args.n_heads,
                      args.d_model,
                      args.d_dff,
                      dropout_rate=args.dropout_rate,
                      tgt_emb_prj_weight_share=args.tgt_emb_prj_weight_share,
                      pe_maxlen=args.pe_maxlen)
    model = Transformer(encoder, decoder)
    print(model)
    model.cuda()
    # optimizer

    learning_rate = CustomSchedule(args.d_model, warmup_steps=args.warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # solver
    solver = Solver(data, model, optimizer, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
