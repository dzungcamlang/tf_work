#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 10:31 
# @Author  : Ming Liu
# @License : Ali Licence
# @Site    :
# @File    : decoder.py 
# @Software: PyCharm Community Edition

import tensorflow as tf

from model.transformer.attention import MultiHeadAttention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, start_id, end_id, n_tgt_vocab, d_word_vec, n_layers, n_head,
                  d_k, d_v, d_model, d_inner, dropout=0.1, tgt_emb_prj_weight_share=True,
                  pe_maxlen=5000):
        super(Decoder, self).__init__()

        self.start_id = start_id # Start of Sentence
        self.end_id = end_id # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_share = tgt_emb_prj_weight_share
        self.pe_maxlen = pe_maxlen

        self.pos_encoding = positional_encoding(pe_maxlen, d_model)

        self.tgt_word_emb = tf.keras.layers.Embedding(n_tgt_vocab, d_model, embeddings_initializer='normal')

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.dec_layers = [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]

        self.final_layer = tf.keras.layers.Dense(num_classes)

    def call(self, decoder_input, encoder_output, training, mask):
        x = decoder_input
        batch_size = tf.shape(x)[0]
        print(x)
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.rsqrt(tf.cast(self.d_model, tf.float32))
        x += tf.cast(self.pos_encoding[:, :seq_len, :], dtype=tf.float32)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, encoder_output, training, mask)

        # x.shape = (batch_size, target_seq_len, target_vocab_size)
        return x
