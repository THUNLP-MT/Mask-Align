#!/usr/bin/env python
# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import socket
import string
import shutil
import argparse
import subprocess
import numpy as np
import thualign.data as data
import thualign.models as models
import thualign.utils as utils
import thualign.utils.alignment as alignment_utils

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test neural alignment models",
        usage="inferrer.py [<args>] [-h | --help]"
    )
    # test args
    parser.add_argument("--gen-weights", action="store_true", help="whether to generate attention weights")
    parser.add_argument("--gen-vizdata", action="store_true", help="whether to generate visualization data")
    parser.add_argument("--test-aer", action="store_true", help="whether to test aer for alignments")

    # configure file
    parser.add_argument("--config", type=str, required=True,
                        help="Provided config file")
    parser.add_argument("--base-config", type=str, help="base config file")
    parser.add_argument("--data-config", type=str, help="data config file")
    parser.add_argument("--model-config", type=str, help="base config file")
    parser.add_argument("--exp", "-e", default='DEFAULT', type=str, help="name of experiments")

    return parser.parse_args()


def load_vocabulary(params):
    params.vocabulary = {
        "source": data.Vocabulary(params.vocab[0]), 
        "target": data.Vocabulary(params.vocab[1])
    }
    return params

def to_cuda(features):
    for key in features:
        features[key] = features[key].cuda()

    return features

def gen_weights(params):
    """Generate attention weights 
    """
    
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
    dist.init_process_group("nccl", init_method=url,
                            rank=0,
                            world_size=1)

    params = load_vocabulary(params)
    checkpoint = getattr(params, "checkpoint", None) or utils.best_checkpoint(params.output)
    # checkpoint = getattr(params, "checkpoint", None) or utils.latest_checkpoint(params.output)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    test_path = params.test_path
    # Create directory and copy files
    if not os.path.exists(test_path):
        print("Making dir: %s" % test_path)
        os.makedirs(test_path)
        params_pattern = os.path.join(params.output, "*.config")
        params_files = glob.glob(params_pattern)

        for name in params_files:
            new_name = name.replace(params.output, test_path)
            shutil.copy(name, new_name)

    if params.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Create model
    with torch.no_grad():

        model = models.get_model(params).cuda()

        if params.half:
            model = model.half()
        
        model.eval()
        print('loading checkpoint: {}'.format(checkpoint))
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

        # dataset = data.get_dataset(params, "infer")

        get_infer_dataset = data.AlignmentPipeline.get_infer_dataset
        dataset = get_infer_dataset(params.test_input, params)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=None)

        iterator = iter(dataset)
        counter = 0
        pad_max = 1024

        # Buffers for synchronization
        results = [0., 0.]
        
        decoder_cross_attn = []
        decoder_self_attn = []
        encoder_self_attn = []
        decoder_layer_out = []
        encoder_layer_out = []
        f_cross_attn = []
        b_cross_attn = []

        require_pred = getattr(params, "require_pred", False)
        if require_pred:
            f_pred = open(os.path.join(test_path, 'pred.txt'), 'w')
        while True:
            try:
                features = next(iterator)
                features = to_cuda(features)
                batch_size = features["source"].shape[0]
            except:
                break

            t = time.time()
            counter += 1

            # Decode
            acc_cnt, all_cnt, state = model.cal_alignment(features)

            t = time.time() - t

            score = 0.0 if all_cnt == 0 else acc_cnt / all_cnt
            print("Finished batch(%d): %.3f (%.3f sec)" % (counter, score, t))

            results[0] += acc_cnt
            results[1] += all_cnt
            
            source_lengths, target_lengths = features["source_mask"].sum(-1).long().tolist(), features["target_mask"].sum(-1).long().tolist()
            for weight, src_len, tgt_len in zip(state['decoder_attn'], source_lengths, target_lengths):
                decoder_cross_attn.append(weight[:, :, :tgt_len, :src_len])

            if 'encoder_self_attn' in state:
                for weight, src_len in zip(state['encoder_self_attn'], source_lengths):
                    encoder_self_attn.append(weight[:, :, :src_len, :src_len])
            if 'decoder_self_attn' in state:
                for weight, tgt_len in zip(state['decoder_self_attn'], target_lengths):
                    decoder_self_attn.append(weight[:, :, :tgt_len, :tgt_len])
            if 'decoder_layer_out' in state:
                for weight in state['decoder_layer_out']:
                    decoder_layer_out.append(weight)
            if 'encoder_layer_out' in state:
                for weight in state['encoder_layer_out']:
                    encoder_layer_out.append(weight)

            if 'f_cross_attn' in state:
                for weight, src_len, tgt_len in zip(state['f_cross_attn'], source_lengths, target_lengths):
                    f_cross_attn.append(weight[:, :, :tgt_len, :src_len])
            if 'b_cross_attn' in state:
                for weight, src_len, tgt_len in zip(state['b_cross_attn'], source_lengths, target_lengths):
                    b_cross_attn.append(weight[:, :, :tgt_len, :src_len])
            
            if require_pred and 'pred' in state:
                for pred_t, tgt_len in zip(state['pred'], target_lengths):
                    pred_t = pred_t[:tgt_len].cpu().tolist()
                    pred_list = [str(params.mapping['target'][p], encoding='utf-8') for p in pred_t]
                    f_pred.write(' '.join(pred_list) + '\n')

        if 'f_cross_attn' not in state and 'b_cross_attn' not in state:
            # unidirectional model
            torch.save(decoder_cross_attn, os.path.join(test_path, 'decoder_cross_attn.pt'))
        if 'encoder_self_attn' in state:
            torch.save(encoder_self_attn, os.path.join(test_path, 'encoder_self_attn.pt'))
        if 'decoder_self_attn' in state:
            torch.save(decoder_self_attn, os.path.join(test_path, 'decoder_self_attn.pt'))
        if 'decoder_layer_out' in state:
            torch.save(decoder_layer_out, os.path.join(test_path, 'decoder_layer_out.pt'))
        if 'encoder_layer_out' in state:
            torch.save(encoder_layer_out, os.path.join(test_path, 'encoder_layer_out.pt'))
        if 'f_cross_attn' in state:
            torch.save(f_cross_attn, os.path.join(test_path, 'f_cross_attn.pt'))
        if 'b_cross_attn' in state:
            torch.save(b_cross_attn, os.path.join(test_path, 'b_cross_attn.pt'))

        score = 0.0 if results[1] == 0 else results[0] / results[1]
        fout = open(os.path.join(test_path, 'eval_res.txt'), 'w')
        fout.write("========= %s =========\nacc_rate: %f" % (test_path, score) + '\n')
        print("acc_rate: %f" % (score))


def gen_align(params):
    """Generate alignment
    """

    src = [l.strip().split() for l in open(params.test_input[0])]
    tgt = [l.strip().split() for l in open(params.test_input[1])]

    try:
        refs, poss = alignment_utils.parse_refs(params.test_ref)
        has_refs = True
    except:
        has_refs = False
        
    weight_path = getattr(params, 'weight_path', None) or params.test_path
    output_path = getattr(params, 'output_path', None) or weight_path

    extract_params = alignment_utils.get_extract_params(params)
    all_data = []

    if getattr(params, "agree_training", False):
        # bidirectional
        # alignment_layer = ['f', 'b', 'gdf', 'soft']
        alignment_layer = ['soft']
        hyps = {} # forward, backward, gdf, soft-extraction
        for k in alignment_layer:
            hyps[k] = []
        f_weights = torch.load(os.path.join(weight_path, 'f_cross_attn.pt'), map_location='cpu')
        b_weights = torch.load(os.path.join(weight_path, 'b_cross_attn.pt'), map_location='cpu')

        for i in range(len(f_weights)):
            src_i, tgt_i, weight_f, weight_b = src[i], tgt[i], f_weights[i], b_weights[i]
            weight_f, weight_b = weight_f[-1].mean(dim=0), weight_b[-1].mean(dim=0)

            if params.gen_vizdata:
                # for alignment visualization
                src_t, tgt_t =  src_i, tgt_i
                data_t = {
                'src': src_t, 
                'tgt': tgt_t,
                'weights': {},
                'metrics': {}
                }
                if has_refs:
                    ref_t, pos_t = refs[i], poss[i]
                    weights_ref = alignment_utils.align_to_weights(ref_t, pos_t, src_t, tgt_t)
                    data_t['weights']['ref'] = weights_ref
                    data_t['ref'] = alignment_utils.ref2str(ref_t, pos_t)

            label = params.label

            if 'f' in alignment_layer:
                # hard forward extraction
                extract_params_t = extract_params.copy()
                extract_params_t['extract_method'] = 't2s'
                extract_params_t['th'] = 0.0
                align_f = alignment_utils.weights_to_align(weight_f, src_i, tgt_i, **extract_params_t)
                hyps['f'].append(align_f)

            if 'b' in alignment_layer:
                # hard backward extraction
                extract_params_t = extract_params.copy()
                extract_params_t['extract_method'] = 's2t'
                extract_params_t['th'] = 0.0
                align_b = alignment_utils.weights_to_align(weight_b, src_i, tgt_i, **extract_params_t)
                hyps['b'].append(align_b)

            if 'gdf' in alignment_layer:
                # gdf
                align_gdf = alignment_utils.grow_diag_final(align_f, align_b, finalize=False)
                hyps['gdf'].append(align_gdf)

            if 'soft' in alignment_layer:
                # soft extraction
                align_soft, weight_final = alignment_utils.bidir_weights_to_align(weight_f, weight_b, src_i, tgt_i, **extract_params)
                hyps['soft'].append(align_soft)

            if params.gen_vizdata:
                if 'f' in alignment_layer:
                    # hard forward extraction
                    weights_align_f = alignment_utils.align_to_weights(align_f, align_f, src_t, tgt_t)
                    data_t['weights'][label+'_f'] = weight_f
                    data_t['weights'][label+'_f.hard'] = weights_align_f

                if 'b' in alignment_layer:
                    # hard backward extraction
                    weights_align_b = alignment_utils.align_to_weights(align_b, align_b, src_t, tgt_t)
                    data_t['weights'][label+'_b'] = weight_b
                    data_t['weights'][label+'_b.hard'] = weights_align_b

                if 'gdf' in alignment_layer:
                    # gdf
                    weights_align_gdf = alignment_utils.align_to_weights(align_gdf, align_gdf, src_t, tgt_t)
                    data_t['weights']['gdf'] = weights_align_gdf

                if 'soft' in alignment_layer:
                    # soft extraction
                    weights_align_soft = alignment_utils.align_to_weights(align_soft, align_soft, src_t, tgt_t)
                    data_t['weights'][label+'.hard'] = weights_align_soft
                    data_t['weights'][label+'_final'] = weight_final
                
                if has_refs:
                    if 'f' in alignment_layer:
                        data_t['metrics'][label+'_f'] = alignment_utils.alignment_metrics([align_f], [ref_t], [pos_t])
                    if 'b' in alignment_layer:
                        data_t['metrics'][label+'_b'] = alignment_utils.alignment_metrics([align_b], [ref_t], [pos_t])
                    if 'gdf' in alignment_layer:
                        data_t['metrics']['gdf'] = alignment_utils.alignment_metrics([align_gdf], [ref_t], [pos_t])
                    if 'soft' in alignment_layer:
                        data_t['metrics'][label] = alignment_utils.alignment_metrics([align_soft], [ref_t], [pos_t])

                all_data.append(data_t)
    else:
        # unidirectional
        alignment_layer = getattr(params, "alignment_layer", [-1])
        hyps = {}
        for j in alignment_layer:
            hyps[j] = []
        all_weights = torch.load(os.path.join(weight_path, 'decoder_cross_attn.pt'), map_location='cpu')
        for i in range(len(all_weights)):
            src_i, tgt_i, weight_i = src[i], tgt[i], all_weights[i]

            if params.gen_vizdata:
                # for alignment visualization
                src_t, tgt_t =  src_i, tgt_i
                if getattr(params, "data_reverse", False):
                    src_t, tgt_t = tgt_t, src_t
                if getattr(params, "src_eos", False) or getattr(params, "tgt_eos", False):
                    eos_tok = '[eos]'
                    if src_t[0].startswith('▁'):
                        eos_tok = '▁' + eos_tok
                    src_t = src_t + [eos_tok]
                    tgt_t = tgt_t + [eos_tok]

                data_t = {
                    'src': src_t, 
                    'tgt': tgt_t,
                    'weights': {},
                    'metrics': {}
                }
                if has_refs:
                    ref_t, pos_t = refs[i], poss[i]
                    weights_ref = alignment_utils.align_to_weights(ref_t, pos_t, src_t, tgt_t)
                    data_t['weights']['ref'] = weights_ref
                    data_t['ref'] = alignment_utils.ref2str(ref_t, pos_t)

            for l in alignment_layer:
                # l = alignment_layer[j]
                if alignment_layer == [-1]:
                    label = params.label
                else:
                    label = params.label + '_' + str(l)
                weight = torch.mean(weight_i[l], dim=0) # ny x nx
                align = alignment_utils.weights_to_align(weight, src_i, tgt_i, **extract_params)
                if getattr(params, "data_reverse", False):
                    align = align.invert()
                    weight = weight.transpose(-1, -2)
                    label = label + '_r'
                hyps[j].append(align)
                
                if params.gen_vizdata:
                    # weight and alignment visualization
                    weights_align = alignment_utils.align_to_weights(align, align, src_t, tgt_t)
                    data_t['weights'][label] = weight
                    data_t['weights'][label+'.hard'] = weights_align
                    if hasattr(params, 'test_ref'):
                        metric = alignment_utils.alignment_metrics([align], [ref_t], [pos_t])
                        data_t['metrics'][label] = metric
            if params.gen_vizdata:
                all_data.append(data_t)

    if params.gen_vizdata:
        torch.save(all_data, os.path.join(output_path, 'alignment_vizdata.pt'))

    if has_refs and params.test_aer:
        fout = open(os.path.join(output_path, 'aer_res.txt'), 'w')
        scores = {}
        for l in alignment_layer:
            hyp = hyps[l]
            a, p, r = alignment_utils.alignment_metrics(hyp, refs, poss)
            output_name = os.path.join(output_path, 'alignment-{}.txt'.format(l))
            align_out = open(output_name, 'w')
            for align in hyp:
                align_out.write(str(align) + '\n')
            print('{}: {:.1f}% ({:.1f}%/{:.1f}%/{})'.format(output_name, a*100, p*100, r*100, sum([len(x) for x in hyp])))
            fout.write('{}: {:.1f}% ({:.1f}%/{:.1f}%/{})'.format(output_name, a*100, p*100, r*100, sum([len(x) for x in hyp])) + '\n')
            scores[l] = a
        fout.write('\n')
        fout.close()
    
        # write down best alignment
        min_key = min(scores, key=scores.get)
        output_name = os.path.join(output_path, 'alignment.txt')
        with open(output_name, 'w') as align_out:
            for align in hyps[min_key]:
                align_out.write(str(align) + '\n')
        return scores
    else:
        # write down alignments extracted from the last alignment_layer
        output_name = os.path.join(output_path, 'alignment.txt')
        with open(output_name, 'w') as align_out:
            if -1 in hyps.keys():
                k = -1
            elif 'soft' in hyps.keys():
                k = 'soft'
            else:
                k = hyps.keys()[0]
            for align in hyps[k]:
                align_out.write(str(align) + '\n')

def merge_dict(src, tgt, keep_common=True):
    res = {}
    for k, v in src.items():
        res[k] = v
    for k, v in tgt.items():
        if k not in res:
            res[k] = v
        elif keep_common:
            src_k = k + '_f'
            tgt_k = k + '_b'
            res[src_k] = res.pop(k)
            res[tgt_k] = v
    return res

def merge(forward_vizdata, backward_vizdata, bialigns):
    res = []
    assert len(forward_vizdata) == len(backward_vizdata) == len(bialigns)
    for f, b, bialign in zip(forward_vizdata, backward_vizdata, bialigns):
        res_t = {}
        assert f['src'] == b['src'] and f['tgt'] == b['tgt']
        res_t['src'] = f['src']
        res_t['tgt'] = f['tgt']
        res_t['weights'] = merge_dict(f['weights'], b['weights'])
        res_t['metrics'] = merge_dict(f['metrics'], b['metrics'])
        res_t['weights']['bidir'] = alignment_utils.align_to_weights(bialign, bialign, f['src'], f['tgt'])

        if 'ref' in f and 'ref' in b:
            assert f['ref'] == b['ref']
            ref_t, pos_t = alignment_utils.parse_ref(f['ref'])
            metric_t = alignment_utils.alignment_metrics([bialign], [ref_t], [pos_t])
            res_t['metrics']['bidir'] = metric_t
        res.append(res_t)
    return res

def main(args):
    exps = args.exp.split(',')
    exp_params = []

    for exp in exps:
        params = utils.Config.read(args.config, base=args.base_config, data=args.data_config, model=args.model_config, exp=exp)
        exp_params.append(params)
        
        base_dir = params.output
        test_path = os.path.join(base_dir, "test")
        params.test_path = test_path
        
        params.gen_weights = args.gen_weights
        params.gen_vizdata = args.gen_vizdata
        params.test_aer = args.test_aer

        if params.gen_weights:
            gen_weights(params)
        gen_align(params)

    if len(exps) == 2:
        import thualign.scripts.combine_bidirectional_alignments
        import thualign.scripts.aer

        def infer_test_path(params):
            weight_path = getattr(params, 'weight_path', None) or params.test_path
            output_path = getattr(params, 'output_path', None) or weight_path
            return output_path

        test_paths = [infer_test_path(params) for params in exp_params]
        common_path = os.path.commonpath(test_paths)
        output_alignment_file = os.path.join(common_path, 'alignment.txt')

        # combine bidirectional alignments
        completedProcess = subprocess.run('python {script} {alignments} --dont_reverse --method grow-diagonal > {output}'.format(
            script=thualign.scripts.combine_bidirectional_alignments.__file__,
            alignments=' '.join([os.path.join(test_path, 'alignment.txt') for test_path in test_paths]),
            output=output_alignment_file
        ), shell=True)

        # calculate aer for combined alignments
        completedProcess = subprocess.run('python {script} {ref} {alignment} > {aer_res}'.format(
            script=thualign.scripts.aer.__file__,
            ref=exp_params[0].test_ref,
            alignment=output_alignment_file,
            aer_res=os.path.join(common_path, 'aer_res.txt')
        ), shell=True)

        print(open(os.path.join(common_path, 'aer_res.txt')).read())

        merge_data_flag = True
        vizdata_files = [os.path.join(test_path, 'alignment_vizdata.pt') for test_path in test_paths]
        for vizdata_file in vizdata_files:
            merge_data_flag = merge_data_flag and os.path.exists(vizdata_file)

        if merge_data_flag:
            forward_vizdata = torch.load(vizdata_files[0], map_location='cpu') # list of ['src', 'tgt', 'ref', 'weights', 'metrics']
            backward_vizdata = torch.load(vizdata_files[1], map_location='cpu') # list of ['src', 'tgt', 'ref', 'weights', 'metrics']
            bialigns = [alignment_utils.parse_ref(line.strip())[0] for line in open(output_alignment_file)]
            merged_vizdata = merge(forward_vizdata, backward_vizdata, bialigns)
            torch.save(merged_vizdata, os.path.join(common_path, 'alignment_vizdata.pt'))
            

    if len(exps) > 2:
        print('More than two experiments are not supported! No merging action will be performed.')

if __name__ == "__main__":
    main(parse_args())