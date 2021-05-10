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

class AttentionSubLayer(modules.Module):

    def __init__(self, params, leaky=False, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout, leaky=leaky)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x, bias, memory=None, state=None, require_weight=False):
        # import ipdb; ipdb.set_trace()
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        if self.training or state is None:
            y = self.attention(y, bias, memory, None, require_weight=require_weight)
            if require_weight:
                y, weights = y
        else:
            kv = [state["k"], state["v"]]
            output = self.attention(y, bias, memory, kv, require_weight=require_weight)
            if require_weight:
                y, k, v, weights = output
            else:
                y, k, v = output
            state["k"], state["v"] = k, v

        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            y = x + y
        else:
            y = self.layer_norm(x + y)

        if require_weight:
            return y, weights
        else:
            return y


class FFNSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(FFNSubLayer, self).__init__(name=name)

        self.dropout = params.residual_dropout
        self.normalization = params.normalization

        with utils.scope(name):
            self.ffn_layer = modules.FeedForward(params.hidden_size,
                                                 params.filter_size,
                                                 dropout=params.relu_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

    def forward(self, x):
        if self.normalization == "before":
            y = self.layer_norm(x)
        else:
            y = x

        y = self.ffn_layer(y)
        y = nn.functional.dropout(y, self.dropout, self.training)

        if self.normalization == "before":
            return x + y
        else:
            return self.layer_norm(x + y)


class TransformerEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params)
            self.feed_forward = FFNSubLayer(params)

    def forward(self, x, bias):
        # import ipdb; ipdb.set_trace()
        x = self.self_attention(x, bias)
        x = self.feed_forward(x)
        return x


class TransformerDecoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(TransformerDecoderLayer, self).__init__(name=name)
        leaky_self_attn = getattr(params, 'leaky_self_attn', False)
        leaky_encdec_attn = getattr(params, 'leaky_encdec_attn', False)

        with utils.scope(name):
            self.self_attention = AttentionSubLayer(params, leaky=leaky_self_attn,
                                                    name="self_attention")
            self.encdec_attention = AttentionSubLayer(params, leaky=leaky_encdec_attn,
                                                    name="encdec_attention")
            self.feed_forward = FFNSubLayer(params)

    def __call__(self, x, attn_bias, encdec_bias, memory):
        x = self.self_attention(x, attn_bias)
        x, weights = self.encdec_attention(x, encdec_bias, memory, require_weight=True)
        x = self.feed_forward(x)
        return x, weights


class TransformerEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(TransformerEncoder, self).__init__(name=name)

        self.normalization = params.normalization

        with utils.scope(name):
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_encoder_layers)])
            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, bias):
        # import ipdb; ipdb.set_trace()
        for layer in self.layers:
            x = layer(x, bias)

        if self.normalization == "before":
            x = self.layer_norm(x)

        return x


class TransformerDecoder(modules.Module):

    def __init__(self, params, name="decoder"):
        super(TransformerDecoder, self).__init__(name=name)

        self.normalization = params.normalization
        self.last_cross = getattr(params, 'last_cross', False)

        with utils.scope(name):
            layer_cls = TransformerEncoderLayer if self.last_cross else TransformerDecoderLayer
            self.layers = nn.ModuleList([layer_cls(params, name="layer_%d" % i) for i in range(params.num_decoder_layers-1)])
            self.layers.append(TransformerDecoderLayer(params, name="layer_%d" % (params.num_decoder_layers-1)))

            if self.normalization == "before":
                self.layer_norm = modules.LayerNorm(params.hidden_size)
            else:
                self.layer_norm = None

    def forward(self, x, attn_bias, encdec_bias, memory):
        all_weights = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoderLayer):
                x = layer(x, attn_bias)
            else:
                x, weights = layer(x, attn_bias, encdec_bias, memory)
                weights = weights.unsqueeze(1) # b x 1 x h x nq x nk
                all_weights.append(weights)

        all_weights = torch.cat(all_weights, dim=1) # b x n_layer x h x nq x nk
        if self.normalization == "before":
            x = self.layer_norm(x)

        return x, all_weights


class TransformerAlign(AlignmentModel):

    def __init__(self, params, name="transformer_align"):
        super(TransformerAlign, self).__init__(params, name=name)

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = TransformerEncoder(params)
            self.decoder = TransformerDecoder(params)

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

        return state

    def decode(self, features, state):
        tgt_seq = features["target"]
        tgt_seq = torch.cat([torch.ones(tgt_seq.shape[0], 1).to(tgt_seq), tgt_seq], dim=1)[:, :-1]

        enc_attn_bias = state["enc_attn_bias"]
        dec_mask = features['target_mask'] # b x n
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = nn.functional.dropout(self.encoding(decoder_input),
                                              self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        decoder_output, all_weights = self.decoder(decoder_input, dec_attn_bias, enc_attn_bias, encoder_output)

        state["decoder_attn"] = all_weights.to(torch.float32)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features):
        if isinstance(features, tuple):
            features, labels = features
        else:
            labels = features["target"]
        mask = features["target_mask"] # b x nq
        state = self.empty_state(features["target"].shape[0],
                                 labels.device)
        state = self.encode(features, state)
        logits, _ = self.decode(features, state)

        loss, log_output = self.cal_loss(logits, labels, mask)
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

    def empty_state(self, batch_size, device):
        # state = {
        #     "decoder": {
        #         "layer_%d" % i: {
        #             "k": torch.zeros([batch_size, 0, self.hidden_size],
        #                              device=device),
        #             "v": torch.zeros([batch_size, 0, self.hidden_size],
        #                              device=device)
        #         } for i in range(self.num_decoder_layers)
        #     }
        # }
        state = {}
        return state

    def cal_alignment(self, features):
        if isinstance(features, tuple):
            features, tgt_seq = features
        else:
            tgt_seq = features["target"]

        tgt_mask = features["target_mask"] # b x nq
        state = self.empty_state(features["target"].shape[0],
                                 tgt_seq.device)
        state = self.encode(features, state)
        logits, state = self.decode(features, state) # b*nq x V

        pred = logits.argmax(-1).view(tgt_seq.shape[0], -1) # b x n
        acc_cnt = ((tgt_seq == pred) * tgt_mask).sum()
        all_cnt = tgt_mask.sum()

        if getattr(self.params, "shift", False):
            state["decoder_attn"] = state["decoder_attn"][:, :, :, 1:, :].to(torch.float32)
        else:
            state["decoder_attn"] = state["decoder_attn"][:, :, :, :-1, :].to(torch.float32)

        state['pred'] = pred
        alignment_layer = getattr(self.params, 'alignment_layer', [2])[-1]
        state['alignment_score'] = state['decoder_attn'][:, alignment_layer].mean(1)

        return acc_cnt, all_cnt, state