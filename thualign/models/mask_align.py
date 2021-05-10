# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thualign.utils as utils
import thualign.modules as modules

from thualign.models.alignment_base import AlignmentModel
from thualign.models.transformer_align import (
    AttentionSubLayer,
    FFNSubLayer,
    TransformerEncoder,
    TransformerEncoderLayer
)

class StaticKVDecoderLayer(TransformerEncoderLayer):

    def forward(self, x, bias, static_kv):
        x = self.self_attention(x, bias, static_kv)
        x = self.feed_forward(x)
        return x

class MaskAlignDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(MaskAlignDecoderLayer, self).__init__(name=name)
        leaky_self_attn = getattr(params, 'leaky_self_attn', False)
        leaky_encdec_attn = getattr(params, 'leaky_encdec_attn', False)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params, leaky=leaky_self_attn,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params, leaky=leaky_encdec_attn,
                                                    name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory, static_kv):
        x = self.self_attention(x, attn_bias, static_kv)
        x, weights = self.encdec_attention(x, encdec_bias, memory, require_weight=True)
        x = self.feed_forward(x)
        return x, weights

class MaskAlignDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(MaskAlignDecoder, self).__init__(name=name)

        self.normalization = params.normalization
        self.last_cross = getattr(params, 'last_cross', False)

        with utils.scope(name):
            layer_cls = StaticKVDecoderLayer if self.last_cross else MaskAlignDecoderLayer
            self.layers = nn.ModuleList([layer_cls(params, name="layer_%d" % i) for i in range(params.num_decoder_layers-1)])
            self.layers.append(MaskAlignDecoderLayer(params, name="layer_%d" % (params.num_decoder_layers-1)))

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, attn_bias, encdec_bias, memory, static_kv):
        all_weights = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, StaticKVDecoderLayer):
                x = layer(x, attn_bias, static_kv)
            else:
                x, weights = layer(x, attn_bias, encdec_bias, memory, static_kv)
                weights = weights.unsqueeze(1) # b x 1 x h x nq x nk
                all_weights.append(weights)

        all_weights = torch.cat(all_weights, dim=1) # b x n_layer x h x nq x nk
        if self.normalization == "before":
            x = self.layer_norm(x)

        return x, all_weights

class MaskAlign(AlignmentModel):

    def __init__(self, params, name="mask_align"):
        super(MaskAlign, self).__init__(params, name=name)

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = TransformerEncoder(params)
            self.decoder = MaskAlignDecoder(params)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.reset_parameters()

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        # import ipdb; ipdb.set_trace()
        return state
    
    def decode(self, features, state):
        tgt_seq = features["target"] # b x n

        b, n = tgt_seq.shape
        enc_attn_bias = state["enc_attn_bias"]
        dec_mask = features['target_mask'] # b x n 
        
        self_mask = (1.0 - torch.eye(n)).expand(b, 1, n, n).to(dec_mask) # b x 1 x n x n
        dec_attn_bias_mask = dec_mask.unsqueeze(1).unsqueeze(1) # b x 1 x 1 x n
        dec_attn_bias_mask = dec_attn_bias_mask.expand(b, 1, n, n) # b x 1 x n x n
        dec_attn_bias_mask = dec_attn_bias_mask * self_mask # b x 1 x n x n
        dec_attn_bias = (1.0 - dec_attn_bias_mask) * (-self.inf) # b x 1 x n x n

        dec_mask = dec_mask.unsqueeze(-1) # b x n x 1
        inputs = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding) # b x n x d
        tqt_emb = inputs * (self.hidden_size ** 0.5)
        inputs = nn.functional.dropout(self.encoding(tqt_emb),               # b x n x d
                                        self.dropout, self.training) 
        static_kv = inputs

        inputs = torch.zeros_like(inputs)
        inputs = nn.functional.dropout(self.encoding(inputs),               # b x n x d
                                    self.dropout, self.training)
            
        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(inputs)

        decoder_output, all_weights = self.decoder(inputs, dec_attn_bias, enc_attn_bias, encoder_output, static_kv) # b x nq x d; b x n_layer x h x nq x nk

        dec_mask = dec_mask.unsqueeze(1).unsqueeze(1) # b x 1 x 1 x nq x 1
        all_weights = all_weights * dec_mask # b x n_layer x h x nq x nk

        state['decoder_attn'] = all_weights

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size]) # b*nq x d

        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1) # b*nq x V

        return logits, state

    def forward(self, features):
        mask = features["target_mask"]
        tgt_seq = features["target"]

        state = {}
        state = self.encode(features, state)
        logits, _ = self.decode(features, state)

        loss, log_output = self.cal_loss(logits, tgt_seq, mask)
        return loss, log_output

    def cal_loss(self, net_output, labels, mask):
        loss = self.criterion(net_output, labels)

        mask = mask.to(torch.float32)

        # Prevent FP16 overflow
        if loss.dtype == torch.float16:
            loss = loss.to(torch.float32)

        loss = torch.sum(loss * mask) / torch.sum(mask)
        loss = loss.to(net_output)
        log_output = "loss: {:.3f}".format(loss)

        return loss, log_output

    def cal_alignment(self, features):
        tgt_seq = features["target"]
        tgt_mask = features["target_mask"] # b x nq
        state = {}
        state = self.encode(features, state)
        logits, state = self.decode(features, state) # b*nq x V

        pred = logits.argmax(-1).view(tgt_seq.shape[0], -1) # b x n
        acc_cnt = ((tgt_seq == pred) * tgt_mask).sum()
        all_cnt = tgt_mask.sum()

        state['pred'] = pred
        state['alignment_score'] = state['decoder_attn'][:, -1].mean(1)

        return acc_cnt, all_cnt, state