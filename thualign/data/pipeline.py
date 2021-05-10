# coding=utf-8
# Copyright 2021-Present The THUAlign Authors

import random
import torch
import numpy as np

from thualign.data.dataset import Dataset, ElementSpec, MapFunc, TextLineDataset
from thualign.data.vocab import Vocabulary
from thualign.tokenizers import WhiteSpaceTokenizer
from typing import Any, Dict, NoReturn, List, Tuple, Union, Callable


def _sort_input_file(filename, reverse=True):
    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, sorted_inputs


class MTPipeline(object):

    @staticmethod
    def get_train_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])
        
        src_dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())
        tgt_dataset = tgt_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())
        lab_dataset = lab_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))


        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq, device="cpu")
            tgt_seq = torch.tensor(tgt_seq, device="cpu")
            labels = torch.tensor(labels, device="cpu")
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float().cpu()
            tgt_mask = tgt_mask.float().cpu()

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_eval_dataset(filenames, params, cpu=False):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])
        lab_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset, lab_dataset))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, tgt_seq, labels = inputs
            src_seq = torch.tensor(src_seq, device="cpu")
            tgt_seq = torch.tensor(tgt_seq, device="cpu")
            labels = torch.tensor(labels, device="cpu")
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float().cpu()
            tgt_mask = tgt_mask.float().cpu()

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return dataset

    @staticmethod
    def get_infer_dataset(filename, params, cpu=False):
        sorted_keys, sorted_data = _sort_input_file(filename)
        src_vocab = params.vocabulary["source"]

        src_dataset = TextLineDataset(sorted_data)
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                    torch.distributed.get_rank())

        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq = torch.tensor(inputs, device="cpu")
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            src_mask = src_mask.float().cpu()

            features = {
                "source": src_seq,
                "source_mask": src_mask,
            }

            return features

        map_obj = MapFunc(map_fn, ElementSpec("Tensor", "{key: [None, None]}"))

        dataset = dataset.map(map_obj)
        dataset = dataset.background()

        return sorted_keys, dataset


class LMPipeline(object):

    @staticmethod
    def get_train_dataset(filename, params):
        vocab = params.vocabulary["source"]

        #src_dataset = TextLineDataset(filename,
        #                              torch.distributed.get_world_size(),
        #                              torch.distributed.get_rank())
        #lab_dataset = TextLineDataset(filename,
        #                              torch.distributed.get_world_size(),
        #                              torch.distributed.get_rank())
        src_dataset = TextLineDataset(filename)
        lab_dataset = TextLineDataset(filename)
        src_dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())
        lab_dataset = lab_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())

        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           params.bos, None)
        lab_dataset = lab_dataset.tokenize(WhiteSpaceTokenizer(),
                                           None, params.eos)
        src_dataset = Dataset.lookup(src_dataset, vocab, vocab[params.unk])
        lab_dataset = Dataset.lookup(lab_dataset, vocab, vocab[params.unk])
        dataset = Dataset.zip((src_dataset, lab_dataset))

        def bucket_boundaries(max_length, min_length=8, step=8):
            l = min_length
            boundaries = []

            while l <= max_length:
                boundaries.append(l + 1)
                l += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            seq, labels = inputs
            seq = np.array(seq)
            labels = np.array(labels)
            mask = seq != params.vocabulary["source"][params.pad]
            mask = mask.astype("float32")

            features = {
                "source": seq,
                "source_mask": mask
            }

            return features, labels

        map_obj = MapFunc(map_fn, ElementSpec("Array", "{key: [None, None]}"))
        dataset = dataset.map(map_obj)

        dataset = dataset.background()

        return dataset


class AlignmentPipeline(object):

    @staticmethod
    def get_train_dataset(filenames, params):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())
        tgt_dataset = tgt_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())

        src_bos = params.bos if getattr(params, "src_bos", False) else None
        src_eos = params.eos if getattr(params, "src_eos", False) else None
        tgt_bos = params.bos if getattr(params, "tgt_bos", False) else None
        tgt_eos = params.eos if getattr(params, "tgt_eos", False) else None
        
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           src_bos, src_eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           tgt_bos, tgt_eos)

        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset))

        def bucket_boundaries(max_length, min_length=8, step=8):
            l = min_length
            boundaries = []

            while l <= max_length:
                boundaries.append(l + 1)
                l += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        dataset = Dataset.bucket_by_sequence_length(
            dataset, boundaries, batch_sizes, pad=src_vocab[params.pad],
            min_length=params.min_length, max_length=params.max_length)

        def map_fn(inputs):
            src_seq, tgt_seq = inputs
            src_seq = torch.tensor(src_seq, device="cpu")
            tgt_seq = torch.tensor(tgt_seq, device="cpu")
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float().cpu()
            tgt_mask = tgt_mask.float().cpu()

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }
            
            return features

        map_obj = MapFunc(map_fn, \
            ElementSpec("Array", "{key: [None, None, None, None]}"))
        dataset = dataset.map(map_obj)
        dataset = dataset.background()
        return dataset
        
    @staticmethod
    def get_infer_dataset(filenames, params):
        src_vocab = params.vocabulary["source"]
        tgt_vocab = params.vocabulary["target"]

        src_dataset = TextLineDataset(filenames[0])
        tgt_dataset = TextLineDataset(filenames[1])

        src_dataset = src_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())
        tgt_dataset = tgt_dataset.shard(torch.distributed.get_world_size(),
                                        torch.distributed.get_rank())

        src_bos = params.bos if getattr(params, "src_bos", False) else None
        src_eos = params.eos if getattr(params, "src_eos", False) else None
        tgt_bos = params.bos if getattr(params, "tgt_bos", False) else None
        tgt_eos = params.eos if getattr(params, "tgt_eos", False) else None
        
        src_dataset = src_dataset.tokenize(WhiteSpaceTokenizer(),
                                           src_bos, src_eos)
        tgt_dataset = tgt_dataset.tokenize(WhiteSpaceTokenizer(),
                                           tgt_bos, tgt_eos)

        src_dataset = Dataset.lookup(src_dataset, src_vocab,
                                     src_vocab[params.unk])
        tgt_dataset = Dataset.lookup(tgt_dataset, tgt_vocab,
                                     tgt_vocab[params.unk])

        dataset = Dataset.zip((src_dataset, tgt_dataset))
        dataset = dataset.padded_batch(params.decode_batch_size,
                                       pad=src_vocab[params.pad])

        def map_fn(inputs):
            src_seq, tgt_seq = inputs
            src_seq = torch.tensor(src_seq, device="cpu")
            tgt_seq = torch.tensor(tgt_seq, device="cpu")
            src_mask = src_seq != params.vocabulary["source"][params.pad]
            tgt_mask = tgt_seq != params.vocabulary["target"][params.pad]
            src_mask = src_mask.float().cpu()
            tgt_mask = tgt_mask.float().cpu()

            features = {
                "source": src_seq,
                "source_mask": src_mask,
                "target": tgt_seq,
                "target_mask": tgt_mask
            }
            
            return features

        map_obj = MapFunc(map_fn, \
            ElementSpec("Array", "{key: [None, None, None, None]}"))
        dataset = dataset.map(map_obj)
        dataset = dataset.background()
        return dataset