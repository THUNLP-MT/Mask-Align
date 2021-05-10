# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import thualign.utils as utils

from thualign.modules.module import Module
from thualign.modules.affine import Affine


class Attention(Module):

    def __init__(self, q_size, k_size, hidden_size, name="attention"):
        super(Attention, self).__init__(name)

        self._q_size = q_size
        self._k_size = k_size
        self._hidden_size = hidden_size

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size, name="q_transform")
            self.k_transform = Affine(k_size, hidden_size, name="k_transform")
            self.v_transform = Affine(hidden_size, 1,
                                      name="v_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, 1]
        logits = torch.transpose(logits, 1, 2)
        # [batch, 1, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.softmax(logits, dim=-1)

        # [batch, 1, length]
        weights = torch.squeeze(weights, 2)
        output = torch.matmul(weights, memory)

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight)
            nn.init.xavier_uniform_(self.k_transform.weight)
            nn.init.xavier_uniform_(self.v_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAttentionBase(Module):

    def __init__(self, name="multihead_attention_base"):
        super(MultiHeadAttentionBase, self).__init__(name=name)

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])


class MultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0, leaky=False,
                 name="multihead_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.leaky = leaky

        with utils.scope(name):
            self.q_transform = Affine(hidden_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(hidden_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")
            if leaky:
                self.leaky_k = nn.Parameter(torch.empty(1, 1, hidden_size))
                self.leaky_v = nn.Parameter(torch.empty(1, 1, hidden_size))
                self.add_name(self.leaky_k, "leaky_k")
                self.add_name(self.leaky_v, "leaky_v")

        self.reset_parameters()

    def forward(self, query, bias, memory=None, kv=None, require_weight=False):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        b, nk, d = k.shape

        # add leaky position, nk' = (nk+1) is self.leaky else nk
        if self.leaky:
            k = torch.cat([self.leaky_k.expand(k.shape[0], -1, -1), k], dim=1) # b x nk' x d
            v = torch.cat([self.leaky_v.expand(v.shape[0], -1, -1), v], dim=1) # b x nk' x d

        # split heads
        qh = self.split_heads(q, self.num_heads) # b x h x nq x d
        kh = self.split_heads(k, self.num_heads) # b x h x nk' x d
        vh = self.split_heads(v, self.num_heads) # b x h x nk' x d

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5 # b x h x nq x d

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            if self.leaky:
                bias = torch.cat([bias.new_zeros(bias.shape[0], 1, 1, 1), bias], dim=-1) # b x 1 x 1 x 1
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)                    # b x h x nq x nk'

        x = torch.matmul(weights, vh)

        if self.leaky:
            # we only consider weights without the leaky postion
            weights = torch.split(weights, [1, nk], dim=-1)[1] # b x h x nq x nk

        # combine heads
        output = self.o_transform(self.combine_heads(x)) # b x nq x d

        if kv is not None:
            if require_weight:
                return output, k, v, weights
            else:
                return output, k, v

        if require_weight:
            return output, weights
        else:
            return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
            if self.leaky:
                nn.init.normal_(self.leaky_k, mean=0.0, std=self.hidden_size ** -0.5)
                nn.init.normal_(self.leaky_v, mean=0.0, std=self.hidden_size ** -0.5)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAdditiveAttention(MultiHeadAttentionBase):

    def __init__(self, q_size, k_size, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAdditiveAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(k_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, num_heads,
                                      name="v_transform")
            self.o_transform = Affine(k_size, k_size,
                                      name="o_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, num_heads]
        logits = torch.transpose(logits, 1, 2)
        # [batch, num_heads, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)

        vh = self.split_heads(memory, self.num_heads)
        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)