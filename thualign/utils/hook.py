# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

import torch
import numpy as np

from .summary import scalar
from .misc import get_global_step

def print_grad(x, name="x"):
    if type(x) == torch.Tensor:
        x.register_hook(lambda x: print("Norm - {} {}:{}\n {}".format(name, list(x.shape), torch.norm(x), x)))
    elif type(x) == torch.nn.Module:
        pass # TODO

def print_grad_norm(x, name="x", summary=True, verbose=True):
    if type(x) == torch.Tensor:
        if verbose:
            x.register_hook(lambda x: print("Norm - {} {}: {}".format(name, list(x.shape), torch.norm(x))))
        if summary:
            scalar('grad_norm/' + name + '/max', torch.max(x), get_global_step(), write_every_n_steps=1)
            scalar('grad_norm/' + name + '/min', torch.min(x), get_global_step(), write_every_n_steps=1)
            scalar('grad_norm/' + name + '/mean', torch.mean(x), get_global_step(), write_every_n_steps=1)
            scalar('grad_norm/' + name + '/normavg', torch.norm(x)/x.nelement(), get_global_step(), write_every_n_steps=1)
    elif type(x) == torch.nn.Module:
        pass # TODO

def print_grad_max(x, name="x"):
    if type(x) == torch.Tensor:
        x.register_hook(lambda x: print("Grad max - {} {}:\n {}".format(name, list(x.shape), torch.max(x).item())))
    elif type(x) == torch.nn.Module:
        pass # TODO

global_collection = {}
collection_on = False

def start_global_collection():
    global collection_on
    collection_on = True

def stop_global_collection():
    global collection_on
    collection_on = False

def add_global_collection(v, name="var"):
    global global_collection
    if not collection_on:
        return
    if name in global_collection:
        global_collection[name].append(v)
    else:
        global_collection[name] = [v]

def get_global_collection(name):
    global global_collection
    if name in global_collection:
        return global_collection[name]
    return None

def clear_global_collection():
    global global_collection
    global_collection = {}