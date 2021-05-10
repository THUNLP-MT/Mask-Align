# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import string
from itertools import product
from nltk.translate import Alignment
import torch

def ref2str(ref, pos):
    return str(ref) + ' ' + str(Alignment(pos-ref)).replace('-', 'p')

def parse_ref(ref_str):
    ref = Alignment.fromstring(re.sub(r'[0-9]*p[0-9]*', "", ref_str))
    pos = Alignment.fromstring(ref_str.replace('p', '-'))
    return ref, pos

def parse_refs(filename):
    refs = []
    poss = []
    for line in open(filename):
        line = line.strip()
        refs.append(Alignment.fromstring(re.sub(r'[0-9]*p[0-9]*', "", line)))
        poss.append(Alignment.fromstring(line.replace('p','-')))
    return refs, poss

def alignment_metrics(hyps, refs, poss):
    n_common_ref = sum([len(hyp & ref) for hyp, ref in zip(hyps, refs)])
    n_common_pos = sum([len(hyp & pos) for hyp, pos in zip(hyps, poss)])
    n_hyps = sum([len(hyp) for hyp in hyps])
    n_refs = sum([len(ref) for ref in refs])
    precision = n_common_pos / float(n_hyps) if n_hyps != 0 else 0
    recall = n_common_ref / float(n_refs)
    aer = 1.0 - (n_common_ref + n_common_pos) / float(n_hyps + n_refs)
    return aer, precision, recall

def merge(tokens):
    n = len(tokens)
    group = []
    res = []
    cnt = -1
    if tokens[0].startswith('▁'):
        # sentencepiece style
        # "▁你 好 ▁啊"
        for i in range(n):
            if tokens[i].startswith('▁'):
                res.append(tokens[i].replace('▁',''))
                cnt += 1
                group.append(cnt)
            else:
                res[-1] = res[-1] + tokens[i]
                group.append(cnt)
    else:
        # subword-nmt style
        # "你@@ 好 啊"
        last_flag = False
        for i in range(n):
            if tokens[i].endswith('@@'):
                cur_flag = True
                tok = tokens[i].replace('@@','')
            else:
                cur_flag = False
                tok = tokens[i]
            if last_flag:
                res[-1] = res[-1] + tok
                group.append(cnt)
            else:
                res.append(tok)
                cnt += 1
                group.append(cnt)
            last_flag = cur_flag
    reverse_group = [[] for i in range(max(group)+1)]
    for i in range(len(group)):
        reverse_group[group[i]].append(i)
    return res, group, reverse_group

def bpe2none(align_list, src, tgt, one_start=False):
    align_t = set()
    src_t, src_group, _ = merge(src)
    tgt_t, tgt_group, _ = merge(tgt)
    for x, y in align_list:
        ax, ay = src_group[x], tgt_group[y]
        if one_start:
            ax, ay = ax + 1, ay + 1
        align_t.add((ax, ay))
    align = Alignment(align_t)
    return align

def none2bpe(align_list, src, tgt, one_start=False):
    align_t = set()
    _, _, src_r = merge(src)
    _, _, tgt_r = merge(tgt)
    for x, y in align_list:
        if one_start:
            x, y = x - 1, y - 1
        for xx, yy in product(src_r[x], tgt_r[y]):
            align_t.add((xx,yy))
    align = Alignment(align_t)
    return align

def align_to_weights(ref, pos, src, tgt, one_start=True):
    """
    Params:
        ref: Alignment
        pos: Alignment
        src: bpe tokens
        tgt: bpe tokens

    Returns:
        weight: list of [x, y, 1/0.5]
    """
    ref = none2bpe(ref, src, tgt, one_start=one_start)
    pos = none2bpe(pos, src, tgt, one_start=one_start)
    t = []
    for x, y in ref & pos:
        t.append([x, y, 1])
    for x, y in pos - ref:
        t.append([x, y, 0.5])
    return t

def get_extract_params(params):
    extract_params = {
        'extract_method': "t2s",
        'th': 0.0
    }

    data_reverse = getattr(params, "data_reverse", False)
    idx = 1 if data_reverse else 0
    if hasattr(params, 'extract_method'):
        if isinstance(params.extract_method, list):
            extract_params['extract_method'] = params.extract_method[idx]
        else:
            extract_params['extract_method'] = params.extract_method
    if hasattr(params, 'extract_th'):
        if isinstance(params.extract_th, list):
            extract_params['th'] = params.extract_th[idx]
        else:
            extract_params['th'] = params.extract_th
    if hasattr(params, 'remove_punc'):
        extract_params['remove_punc'] = params.remove_punc
    if hasattr(params, 'src_eos'):
        extract_params['src_eos'] = params.src_eos
    if hasattr(params, 'tgt_eos'):
        extract_params['tgt_eos'] = params.tgt_eos
    return extract_params

def clean_weights(weights, src, tgt, src_eos=False, tgt_eos=False, remove_punc=True):
    if src_eos:
        weights = weights[:, :-1]
    if tgt_eos:
        weights = weights[:-1, :]

    if remove_punc:
        weights = weights.clone()
        if src[-1].replace('▁', '') in string.punctuation:
            if tgt[-1].replace('▁', '') in string.punctuation:
                weights[:-1, -1] = 0.0
                weights[-1, -1] = 1.0
            else:
                weights[:, -1] = 0.0
    return weights

def weights_to_align(weights, src, tgt, extract_method='t2s', th=0.0, remove_punc=True, one_start=True, src_eos=False, tgt_eos=False, remove_bpe=True):
    """
    weights: ny x nx; tgt x src
    src: bpe tokens
    tgt: bpe tokens
    """
    weights = clean_weights(weights, src, tgt, src_eos=src_eos, tgt_eos=tgt_eos, remove_punc=remove_punc)
    if weights.shape[0] == 0:
        align_str = '1-1' if one_start else '0-0'
        return Alignment.fromstring(align_str)
    if extract_method == 't2s':
         # tgt -> src
        values, src_indices = weights.max(-1)
        align_list = list(zip(range(src_indices.shape[0]), src_indices.tolist()))
        align_list = [a for v, a in zip(values, align_list) if v.item() > th]
    elif extract_method == 's2t':
        # src -> tgt
        values, tgt_indices = weights.max(-2)
        align_list = list(zip(tgt_indices.tolist(), range(tgt_indices.shape[0])))
        align_list = [a for v, a in zip(values, align_list) if v.item() > th]
    elif extract_method == 'threshold':
        # threshold
        align = weights > th
        align_list = align.nonzero().tolist()
    elif extract_method == 'topk':
        weights = weights.view(-1) # ny * nx
        n = weights.shape[-1]
        values, indices = weights.topk(n)
        align_list = [(int(indice.item()/len(src)), int(indice.item() % len(src))) for indice in indices]
        align_list = [a for v, a in zip(values, align_list) if v.item() > th]

    align_list = [(x, y) for y, x in align_list]

    if remove_bpe:
        align = bpe2none(align_list, src, tgt, one_start=one_start)
    else:
        if one_start:
            align_list = [(x+1, y+1) for (x,y) in align_list]
        align = Alignment(align_list)

    if len(align) == 0:
        align_str = '1-1' if one_start else '0-0'
        align = Alignment.fromstring(align_str)
    return align

def bidir_weights_to_align(weight_f, weight_b, src, tgt, extract_method='topk', th=0.0, remove_punc=False, one_start=True, src_eos=False, tgt_eos=False):
    weight_f = clean_weights(weight_f, src, tgt, src_eos=src_eos, tgt_eos=tgt_eos, remove_punc=remove_punc)
    weight_b = clean_weights(weight_b, src, tgt, src_eos=tgt_eos, tgt_eos=src_eos, remove_punc=remove_punc)
    if weight_f.shape[0] != weight_b.shape[0]:
        weight_b = weight_b.transpose(-1,-2)
    assert weight_f.shape == weight_b.shape

    weight_final = 2*(weight_f * weight_b)/(weight_f + weight_b)
    union = weight_final.view(-1)
    k = union.shape[-1]
    values, indices = union.topk(k)
    align_list = [(index % len(src), index // len(src)) for index in indices]
    align_list = [a for v, a in zip(values, align_list) if v.item() > th]
    align_list = Alignment(align_list)
    
    align = bpe2none(align_list, src, tgt, one_start=one_start)
    
    return align, weight_final


NEIGHBORING = {(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)}

def get_length(align_union):
    """ Estimate length of source and target segment """
    max_e = max((e for e, f in align_union))
    max_f = max((f for e, f in align_union))
    return max_e + 1, max_f + 1

def grow_diag_final(e2f, f2e, finalize=True):
    """ Implemented as in http://www.statmt.org/moses/?n=FactoredTraining.AlignWords """
    e2f, f2e = set(e2f), set(f2e)
    alignments = e2f.intersection(f2e)
    alignment_union = e2f.union(f2e)

    e_len, f_len = get_length(alignment_union)
    alignments = grow_diag(alignments, alignment_union, e_len, f_len)
    if finalize:
        alignments = final(alignments, e2f, e_len, f_len)
        alignments = final(alignments, f2e, e_len, f_len)

    return Alignment(alignments)

def grow_diag(alignments, alignment_union, e_len, f_len):
    """ Adds alignment in the neighborhood of alignments in the intersection """
    finished = False

    while not finished:
        finished = True

        for e, f in product(range(e_len), range(f_len)):
            if (e, f) in alignments:
                for e_new, f_new in ((e + e_delta, f + f_delta) for e_delta, f_delta in NEIGHBORING):
                    if e_new not in {e for e, f in alignments} and f_new not in {f for e, f in alignments} \
                            and (e_new, f_new) in alignment_union:
                        alignments.add((e_new, f_new))
                        finished = False

    return alignments


def final(alignments, directional_alignment, e_len, f_len):
    """ Adds alignments from directional alignment when word is not a valid alignment yet """
    for e_new, f_new in product(range(e_len), range(f_len)):
        if e_new not in {e for e, f in alignments} and f_new not in {f for e, f in alignments} \
                and (e_new, f_new) in directional_alignment:
            alignments.add((e_new, f_new))
    return alignments


def draw_weights(weights, src, tgt):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    weights = weights[:len(tgt), :len(src)]
    # weights_sum = weights.sum(dim=-1)
    # if weights.sum(dim=-1).sum().item() != len(weights):
    #     weights = torch.cat([(1-weights.sum(-1)).unsqueeze(-1), weights], dim=-1)
    #     src = ['[NULL]'] + src
    df = pd.DataFrame(weights.numpy())
    df.columns = src
    df.columns.name = 'src'
    df['tgt'] = tgt
    df = df.set_index('tgt')
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(df, cmap="GnBu", linewidths=0.5, vmin=0, vmax=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8) 
    ax.xaxis.tick_top()
    fig = ax.get_figure()
    return fig