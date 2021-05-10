# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import torch


def oldest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    oldest_counter = 10000000
    checkpoint_name = names[0]

    for name in names:
        counter = name.rstrip(".pt").split("-")[-1]

        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter < oldest_counter:
            checkpoint_name = name
            oldest_counter = counter

    return checkpoint_name


def latest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    latest_counter = 0
    checkpoint_name = names[0]

    for name in names:
        counter = name.rstrip(".pt").split("-")[-1]

        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter > latest_counter:
            checkpoint_name = name
            latest_counter = counter

    return checkpoint_name

def best_checkpoint(path):
    eval_dir = os.path.join(path, 'eval')
    if os.path.exists(eval_dir):
        record_file = os.path.join(eval_dir, 'record')
        if os.path.exists(record_file):
            with open(record_file) as f:
                record = [line.split(':') for line in f]
                record = [(x.replace('"', ''),  float(y)) for x, y in record]
            record = sorted(record, key=lambda x: x[1])
            return os.path.join(eval_dir, record[-1][0])
        else:
            return latest_checkpoint(path)
    else:
        return latest_checkpoint(path)

def save(state, path, max_to_keep=None):
    checkpoints = glob.glob(os.path.join(path, "*.pt"))

    if not checkpoints:
        counter = 1
    else:
        checkpoint = latest_checkpoint(path)
        counter = int(checkpoint.rstrip(".pt").split("-")[-1]) + 1

    if max_to_keep and len(checkpoints) >= max_to_keep:
        checkpoint = oldest_checkpoint(path)
        os.remove(checkpoint)

    checkpoint = os.path.join(path, "model-%d.pt" % counter)
    print("Saving checkpoint: %s" % checkpoint)
    torch.save(state, checkpoint)
