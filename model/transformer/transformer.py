#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 15:01 
# @Author  : Ming Liu
# @License : Ali Licence
# @Site    :
# @File    : transformer.py 
# @Software: PyCharm Community Edition

import tensorflow as tf

from model.transformer.encoder import *
from model.transformer.decoder import *


class Transformer(tf.keras.Model):
    """An encoder-decoder framework only includes attention."""

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, input_lengths, padded_target):
        """
        :param input: N x Ti x D
        :param input_lengths: N
        :param padded_target: N x To
        """
        enc_output = self.encoder(input, input_lengths)  # (batch_size, inp_seq_len, d_model)

        pred, gold = self.decoder(enc_output, padded_target, input_lengths)

        return pred, gold

    def regonize(self, input, input_length, char_list, args):
        """
           Sequence-to-Sequence beam search, decode one utterence now.
        :param input: T x D
        :param input_length:
        :param char_list: list of characters
        :param args: args.beam
        :return: nbest_hyps
        """
        enc_output = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(enc_output[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_output, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_output[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = tf.load(path, map_location=lambda storage, loc: storage)
        model, LFR_m, LFR_n = cls.load_model_from_package(package)
        return model, LFR_m, LFR_n

    @classmethod
    def load_model_from_package(cls, package):
        encoder = Encoder(package['n_layers_enc'],
                          package['n_heads'],
                          package['d_model'],
                          package['d_dff'],
                          pe_maxlen=package['pe_maxlen'],
                          dropout_rate=package['dropout_rate'],)
        decoder = Decoder(package['sos_id'],
                          package['eos_id'],
                          package['vocab_size'],
                          package['d_word_vec'],
                          package['n_layers_dec'],
                          package['n_heads'],
                          package['d_model'],
                          package['d_dff'],
                          dropout_rate=package['dropout'],
                          tgt_emb_prj_weight_share=package['tgt_emb_prj_weight_share'],
                          pe_maxlen=package['pe_maxlen'],
                          )
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        LFR_m, LFR_n = package['LFR_m'], package['LFR_n']
        return model, LFR_m, LFR_n

    @staticmethod
    def serialize(model, optimizer, epoch, LFR_m, LFR_n, tr_loss=None, cv_loss=None):
        package = {
            # Low Frame Rate Feature
            'LFR_m': LFR_m,
            'LFR_n': LFR_n,
            # encoder
            'n_layers_enc': model.encoder.n_layers,
            'n_heads': model.encoder.n_head,
            'd_model': model.encoder.d_model,
            'd_dff': model.encoder.d_dff,
            'dropout_rate': model.encoder.dropout_rate,
            'pe_maxlen': model.encoder.pe_maxlen,
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_share': model.decoder.tgt_emb_prj_weight_share,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

