#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 11:00 
# @Author  : Ming Liu
# @License : Ali Licence
# @Site    :
# @File    : encoder.py 
# @Software: PyCharm Community Edition

import tensorflow as tf

from model.transformer.attention import MultiHeadAttention
from model.transformer.module import *


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_heads, d_model, d_dff, pe_maxlen=5000, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.n_head = n_heads
        self.d_model = d_model
        self.d_inner = d_dff
        self.pe_maxlen = pe_maxlen
        self.dropout_rate = dropout_rate

        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

        self.pos_encoding = positional_encoding(pe_maxlen, d_model)

        self.enc_layers = [EncoderLayer(d_model, d_dff, n_heads, dropout_rate)
                           for _ in range(n_layers)]

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, padded_input, input_len, return_attns=False):
        """
        :param padded_input: [batch_size, input_seq_len, ]
        :param input_len: batch_size
        :param return_attns: Fasle

        :return:
                enc_output: [batch_size, input_seq_len, d_model]
        """
        enc_att_list = []

        # Prepare mask
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_len)
        input_seq_len = tf.shape(padded_input)[1]
        slf_att_mask = get_att_pad_mask(padded_input, input_len, input_seq_len)

        # Encoder
        enc_input = tf.cast(padded_input, dtype=tf.float32)

        enc_input = self.dropout(self.layer_norm(self.dense(enc_input) + self.pos_encoding[:, input_seq_len, :]))

        for i in range(self.n_layers):
            enc_output, enc_slf_att = self.enc_layers[i](enc_input,
                                                         non_pad_mask=non_pad_mask,
                                                         slf_att_mask=slf_att_mask)
            if return_attns:
                enc_att_list += [enc_slf_att]

        if return_attns:
            return enc_output, enc_att_list

        return enc_output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_dff, n_heads, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, enc_input, non_pad_mask=None, slf_att_mask=None):

        # (batch_size, input_seq_len, d_model)
        attn_output, enc_slf_att = self.mha(enc_input, enc_input, enc_input, mask=slf_att_mask)
        attn_output = self.dropout1(attn_output)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(enc_input + attn_output)
        out1 *= non_pad_mask

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)
        out2 *= non_pad_mask

        return out2
