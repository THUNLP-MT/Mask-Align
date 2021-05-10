# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import configparser
import copy
import glob
import logging
import os
import re
import six
import socket
import time
import string
import torch

import thualign.data as data
import torch.distributed as dist
import thualign.models as models
import thualign.optimizers as optimizers
import thualign.utils as utils
import thualign.utils.summary as summary
import thualign.utils.alignment as alignment_utils

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train neural alignment models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training mode")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed precision training")

    # configure file
    parser.add_argument("--config", type=str, required=True,
                        help="Provided config file")
    parser.add_argument("--base-config", type=str, help="base config file")
    parser.add_argument("--data-config", type=str, help="data config file")
    parser.add_argument("--model-config", type=str, help="base config file")
    parser.add_argument("--exp", default='DEFAULT', type=str, help="name of experiments")

    return parser.parse_args(args)

def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    filename = os.path.join(model_dir, "params.config")

    if os.path.exists(filename):
        params.parse_config(filename)

    return params


def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    params.export(filename)

def load_vocabulary(params):
    params.vocabulary = {
        "source": data.Vocabulary(params.vocab[0]), 
        "target": data.Vocabulary(params.vocab[1])
    }
    return params

def print_variables(model, pattern, log=True):
    flags = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name):
            flags.append(True)
        else:
            flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable variables size: %d" % total_size)

    return flags


def exclude_variables(flags, grads_and_vars):
    idx = 0
    new_grads = []
    new_vars = []

    for grad, (name, var) in grads_and_vars:
        if flags[idx]:
            new_grads.append(grad)
            new_vars.append((name, var))

        idx += 1

    return zip(new_grads, new_vars)


def save_checkpoint(step, epoch, model, optimizer, params):
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def get_learning_rate_schedule(params):
    if params.learning_rate_schedule == "linear_warmup_rsqrt_decay":
        schedule = optimizers.LinearWarmupRsqrtDecay(
            params.learning_rate, params.warmup_steps,
            initial_learning_rate=params.initial_learning_rate,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "piecewise_constant_decay":
        schedule = optimizers.PiecewiseConstantDecay(
            params.learning_rate_boundaries, params.learning_rate_values,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "linear_exponential_decay":
        schedule = optimizers.LinearExponentialDecay(
            params.learning_rate, params.warmup_steps,
            params.start_decay_step, params.end_decay_step,
            dist.get_world_size(), summary=params.save_summary)
    elif params.learning_rate_schedule == "constant":
        schedule = params.learning_rate
    else:
        raise ValueError("Unknown schedule %s" % params.learning_rate_schedule)

    return schedule


def get_clipper(params):
    if params.clipping.lower() == "none":
        clipper = None
    elif params.clipping.lower() == "adaptive":
        clipper = optimizers.adaptive_clipper(0.95)
    elif params.clipping.lower() == "global_norm":
        clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
    else:
        raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_optimizer(params, schedule, clipper):
    if params.optimizer.lower() == "adam":
        optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                             beta_1=params.adam_beta1,
                                             beta_2=params.adam_beta2,
                                             epsilon=params.adam_epsilon,
                                             clipper=clipper,
                                             summaries=params.save_summary)
    elif params.optimizer.lower() == "adadelta":
        optimizer = optimizers.AdadeltaOptimizer(
            learning_rate=schedule, rho=params.adadelta_rho,
            epsilon=params.adadelta_epsilon, clipper=clipper,
            summaries=params.save_summary)
    elif params.optimizer.lower() == "sgd":
        optimizer = optimizers.SGDOptimizer(
            learning_rate=schedule, clipper=clipper,
            summaries=params.save_summary)
    else:
        raise ValueError("Unknown optimizer %s" % params.optimizer)

    return optimizer


def load_references(pattern):
    if not pattern:
        return None

    files = glob.glob(pattern)
    references = []

    for name in files:
        ref = []
        with open(name, "rb") as fd:
            for line in fd:
                items = line.strip().split()
                ref.append(items)
        references.append(ref)

    return list(zip(*references))

def to_cuda(features):
    for key in features:
        features[key] = features[key].cuda()

    return features

def main(args):

    params = utils.Config.read(args.config, base=args.base_config, data=args.data_config, model=args.model_config, exp=args.exp)
    params = load_vocabulary(params)

    # Initialize distributed utility
    if args.distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Export parameters
    if dist.get_rank() == 0:
        export_params(params.output, "params.config", params)

    model = models.get_model(params).cuda()

    if getattr(params, "half", False):
        model = model.half()
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model.train()

    # Init tensorboard
    summary.init(params.output, params.save_summary)

    schedule = get_learning_rate_schedule(params)
    clipper = get_clipper(params)
    optimizer = get_optimizer(params, schedule, clipper)

    if getattr(params, "half", False):
        optimizer = optimizers.LossScalingOptimizer(optimizer)

    optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

    if dist.get_rank() == 0:
        print(params)
        print(model)

    trainable_flags = print_variables(model, "",
                                      dist.get_rank() == 0)

    # Train Dataset
    dataset = data.AlignmentPipeline.get_train_dataset(params.train_input, params)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=None)

    # Eval Dataset
    eval_dataset = data.AlignmentPipeline.get_infer_dataset(params.valid_input, params)
    eval_dataset = torch.utils.data.DataLoader(eval_dataset, batch_size=None)

    # Load checkpoint
    checkpoint = utils.latest_checkpoint(params.output)

    if getattr(params, "checkpoint", None) is not None:
        # Load pre-trained models
        state = torch.load(params.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        step = params.initial_step
        epoch = 0
        broadcast(model)
    elif checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0
        epoch = 0
        broadcast(model)

    counter = 0

    while True:
        for features in dataset:
            if counter % params.update_cycle == 0:
                step += 1
                utils.set_global_step(step)

            counter += 1
            t = time.time()
            features = to_cuda(features)
            loss, log_info = model(features)
            gradients = optimizer.compute_gradients(loss,
                                                    list(model.parameters()))
            grads_and_vars = exclude_variables(
                trainable_flags,
                zip(gradients, list(model.named_parameters())))
            optimizer.apply_gradients(grads_and_vars)

            t = time.time() - t

            summary.scalar("loss", loss, step, write_every_n_steps=1)
            summary.scalar("global_step/sec", t, step)

            if dist.get_rank() == 0 and step % params.log_interval == 0 and counter % params.update_cycle == 0:
                print("epoch = %d, step = %d, %s (%.3f sec)" %
                    (epoch + 1, step, log_info, t))

            if counter % params.update_cycle == 0:
                if step >= params.train_steps:
                    utils.evaluate(model, eval_dataset,
                                   params.output, params)
                    save_checkpoint(step, epoch, model, optimizer, params)

                    if dist.get_rank() == 0:
                        summary.close()

                    return

                if step % params.eval_steps == 0:
                    utils.evaluate(model, eval_dataset,
                                   params.output, params)

                if step % params.save_checkpoint_steps == 0:
                    save_checkpoint(step, epoch, model, optimizer, params)

        epoch += 1


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        params = utils.Config.read(parsed_args.config, base=parsed_args.base_config, data=parsed_args.data_config, model=parsed_args.model_config, exp=parsed_args.exp)
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        world_size = len(params.device_list)

        if world_size > 1:
            torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                        nprocs=world_size)
        else:
            process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
