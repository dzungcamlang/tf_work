#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/19 10:31 
# @Author  : Ming Liu
# @License : Ali Licence
# @Site    :
# @File    : decoder.py 
# @Software: PyCharm Community Edition

import tensorflow as tf

from util.data import *
from model.transformer.module import *
from model.transformer.attention import MultiHeadAttention


class Decoder(tf.keras.layers.Layer):

    def __init__(self, start_id, end_id, n_tgt_vocab, d_word_vec, n_layers, n_heads,
                 d_model, d_dff, dropout_rate=0.1, tgt_emb_prj_weight_share=True, pe_maxlen=5000):
        super(Decoder, self).__init__()

        self.sos_id = start_id # Start of Sentence
        self.eos_id = end_id # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_dff = d_dff
        self.dropout_rate = dropout_rate
        self.tgt_emb_prj_weight_share = tgt_emb_prj_weight_share
        self.pe_maxlen = pe_maxlen

        self.pos_encoding = positional_encoding(pe_maxlen, d_model)

        self.dense = tf.keras.layers.Dense(d_model, activation='relu')

        self.tgt_word_emb = tf.keras.layers.Embedding(n_tgt_vocab, d_model, embeddings_initializer='normal')

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dec_layers = [DecoderLayer(d_model, d_dff, n_heads, dropout_rate=dropout_rate) for _ in range(n_layers)]

        # self.final_layer = tf.keras.layers.Dense(num_classes)

        if tgt_emb_prj_weight_share:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1

    def preprocess(self, decoder_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        enc = [en[en != IGNORE_ID] for en in decoder_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = enc[0].new([self.eos_id])
        sos = enc[0].new([self.sos_id])
        enc_in = [tf.concat([sos, x], 0) for x in enc]
        enc_out = [tf.concat([x, eos], 0) for x in enc]
        # padding for ys with -1
        # pys: utt x olen
        enc_in_pad = pad_list(enc_in, self.eos_id)
        enc_out_pad = pad_list(enc_out, IGNORE_ID)
        assert enc_in_pad.size() == enc_out_pad.size()
        return enc_in_pad, enc_out_pad

    def call(self, decoder_input, encoder_output, encoder_input_lengths, return_attn=False):
        """
        :param decoder_input: N x T
        :param encoder_output: N x T x H
        """
        dec_slf_att_list = []
        dec_enc_att_list = []

        # Decoder Input and Output
        enc_in_pad, enc_out_pad = self.preprocess(decoder_input)

        # Prepare mask
        non_pad_mask = get_non_pad_mask(enc_in_pad, pad_idx=self.eos_id)
        slf_att_mask_subseq = get_subsequent_mask(enc_in_pad)
        slf_att_mask_keypad = get_att_key_pad_mask(seq_k=enc_in_pad,
                                                   seq_q=enc_in_pad,
                                                   pad_idx=self.eos_id)
        slf_att_mask = (slf_att_mask_keypad + slf_att_mask_subseq).gt(0)

        output_len = tf.shape(enc_in_pad)[1]
        dec_enc_att_mask = get_att_pad_mask(encoder_output,
                                            encoder_input_lengths,
                                            output_len)

        # Network
        dec_input = self.dropout(self.tgt_word_emb(enc_in_pad) * self.x_logit_scale +
                                  self.positional_encoding(enc_in_pad))

        for i in range(self.n_layers):
            dec_output, dec_slf_att, dec_enc_att = self.dec_layers[i](dec_input,
                                                                      encoder_output,
                                                                      non_pad_mask=non_pad_mask,
                                                                      slf_att_mask=slf_att_mask,
                                                                      dec_enc_att_mask=dec_enc_att_mask)

            if return_attn:
                dec_slf_att_list += [dec_slf_att]
                dec_enc_att_list += [dec_enc_att]

        seq_logits = self.tgt_word_prj(dec_output)

        if return_attn:
            return seq_logits, enc_out_pad, dec_slf_att_list, dec_enc_att_list

        return seq_logits, enc_out_pad

    def recognize_beam(self, encoder_output, char_list, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_output: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        if args.decode_max_len == 0:
            maxlen = tf.shape(encoder_output)[0]
        else:
            maxlen = args.decode_max_len

        encoder_outputs = encoder_output.unsqueeze(0)

        # prepare sos
        ys = torch.ones(1, 1).fill_(self.sos_id).type_as(encoder_outputs).long()

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i

                # Prepare masks
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                slf_att_mask = get_subsequent_mask(ys)

                # Network
                dec_input = self.dropout(
                    self.tgt_word_emb(ys) * self.x_logit_scale +
                    self.positional_encoding(ys))

                for i in range(self.n_layers):
                    dec_output, _, _ = self.dec_layers[i](dec_input,
                                                          encoder_output,
                                                          non_pad_mask=non_pad_mask,
                                                          slf_att_mask=slf_att_mask,
                                                          dec_enc_att_mask=None)

                seq_logit = self.tgt_word_prj(dec_output[:, -1])

                local_scores = F.log_softmax(seq_logit, dim=1)
                # topk scores
                local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = torch.ones(1, (1+ys.size(1))).type_as(encoder_outputs).long()
                    new_hyp['yseq'][:, :ys.size(1)] = hyp['yseq']
                    new_hyp['yseq'][:, ys.size(1)] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)


                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'] = torch.cat([hyp['yseq'],
                                             torch.ones(1, 1).fill_(self.eos_id).type_as(encoder_outputs).long()], dim=1)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, -1] == self.eos_id:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                print('remeined hypothes: ' + str(len(hyps)))
            else:
                print('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                print('hypo: ' + ''.join([char_list[int(x)]
                                          for x in hyp['yseq'][0, 1:]]))
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            :min(len(ended_hyps), nbest)]
        # compitable with LAS implementation
        for hyp in nbest_hyps:
            hyp['yseq'] = hyp['yseq'][0].cpu().numpy().tolist()
        return nbest_hyps


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_dff, n_heads, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.slf_att = MultiHeadAttention(d_model, n_heads)
        self.enc_att = MultiHeadAttention(d_model, n_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, dec_input, enc_output, non_pad_mask=None, slf_att_mask=None, dec_enc_att_mask=None):

        inputs = dec_input
        attn1, dec_slf_att = self.mha1(inputs, inputs, inputs, mask=slf_att_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        attn1 *= non_pad_mask
        out1 = self.layernorm1(attn1 + inputs)

        attn2, dec_enc_att = self.mha2(out1, enc_output, enc_output, mask=dec_enc_att_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        attn2 *= non_pad_mask
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        ffn_output *= non_pad_mask

        dec_output = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return dec_output, dec_slf_att, dec_enc_att
