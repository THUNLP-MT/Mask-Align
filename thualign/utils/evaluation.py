# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import glob
import operator
import os
import shutil
import time
import torch

import torch.distributed as dist

from thualign.data import idxs2str 
from thualign.utils.checkpoint import save, latest_checkpoint
from thualign.utils.inference import beam_search
from thualign.utils.bleu import bleu
from thualign.utils.bpe import BPE
from thualign.utils.misc import get_global_step
from thualign.utils.summary import scalar, figure
from thualign.utils.alignment import draw_weights

def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not os.path.exists(filename):
        return records

    with open(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1].rstrip(".pt"))
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with open(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records

def to_cuda(features):
    for key in features:
        features[key] = features[key].cuda()

    return features

def evaluate(model, dataset, base_dir, params):
    base_dir = base_dir.rstrip("/")
    save_path = os.path.join(base_dir, "eval")
    record_name = os.path.join(save_path, "record")
    log_name = os.path.join(save_path, "log")
    max_to_keep = params.keep_top_checkpoint_max

    if dist.get_rank() == 0:
        # Create directory and copy files
        if not os.path.exists(save_path):
            print("Making dir: %s" % save_path)
            os.makedirs(save_path)

            params_pattern = os.path.join(base_dir, "*.config")
            params_files = glob.glob(params_pattern)

            for name in params_files:
                new_name = name.replace(base_dir, save_path)
                shutil.copy(name, new_name)

    # Do validation here
    global_step = get_global_step()

    if dist.get_rank() == 0:
        print("Validating model at step %d" % global_step)
    
    with torch.no_grad():
        model.eval()
        iterator = iter(dataset)
        counter = 0
        pad_max = 1024

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        results = [0., 0.]

        eval_plot = getattr(params, 'eval_plot', False)
        if dist.get_rank() == 0 and eval_plot:
            fig_features = None

        while True:
            try:
                features = next(iterator)
                features = to_cuda(features)
                batch_size = features["source"].shape[0]
                if dist.get_rank() == 0 and eval_plot and fig_features is None:
                    fig_features = features
            except:
                features = {
                    "source": torch.ones([1, 4]).long(),
                    "source_mask": torch.ones([1, 4]).float(),
                    "target": torch.ones([1, 4]).long(),
                    "target_mask": torch.ones([1, 4]).float(),
                }
                batch_size = 0

            t = time.time()
            counter += 1
            
            acc_cnt, all_cnt, all_weights = model.cal_alignment(features)

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(acc_cnt)
            dist.all_reduce(all_cnt)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            t = time.time() - t

            results[0] += acc_cnt
            results[1] += all_cnt
            score = 0.0
            if all_cnt != 0:
                score = float(acc_cnt) / all_cnt

            info = "{:.3f}".format(score)
            print("Finished batch(%d): %s (%.3f sec)" % (counter, info, t))

        if dist.get_rank() == 0 and eval_plot:
            acc_cnt, all_cnt, state = model.cal_alignment(fig_features) # b x n_layer x h x nq x nk
            alignment_score = state['alignment_score'].cpu()
            src_seqs = idxs2str(fig_features['source'].cpu(), params.vocabulary['source'])
            tgt_seqs = idxs2str(fig_features['target'].cpu(), params.vocabulary['target'])
            cnt = 0
            for align_score, src_seq, tgt_seq in zip(alignment_score, src_seqs, tgt_seqs):
                if len(src_seq) < 20 and len(tgt_seq) < 20:
                    fig = draw_weights(align_score, src_seq, tgt_seq)
                    figure("eval_plot_{}".format(cnt), fig, global_step, write_every_n_steps=1)
                    cnt += 1
                if cnt >= 5:
                    break

    model.train()

    if dist.get_rank() == 0:
        score = 0 if results[1] == 0 else results[0] / results[1]
        scalar("score", score, global_step, write_every_n_steps=1)
        print("score at step %d: %f" % (global_step, score))

        # Save checkpoint to save_path
        save({"model": model.state_dict(), "step": global_step}, save_path)

        _save_log(log_name, ("acc_rate", global_step, score))
        records = _read_score_record(record_name)
        record = [latest_checkpoint(save_path).split("/")[-1], score]

        added, removed, records = _add_to_record(records, record, max_to_keep)

        if added is None:
            # Remove latest checkpoint
            filename = latest_checkpoint(save_path)
            print("Removing %s" % filename)
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        if removed is not None:
            filename = os.path.join(save_path, removed)
            print("Removing %s" % filename)
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        _save_score_record(record_name, records)

        best_score = records[0][1]
        print("Best score at step %d: %f" % (global_step, best_score))