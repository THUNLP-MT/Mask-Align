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

class AgreementWrapper(AlignmentModel):

    def __init__(self, params, model_cls, name="dual_model"):
        super(AgreementWrapper, self).__init__(params, name=name)
        
        with utils.scope(name):
            self.f_model = model_cls(params)
            self.b_model = model_cls(params)

            self.b_model.source_embedding, self.b_model.target_embedding = self.b_model.target_embedding, self.b_model.source_embedding

            model_dir = getattr(params, "pretrained_model_dir", None)
            if model_dir:
                # load pretrained models
                f_ckpt = utils.best_checkpoint(os.path.join(model_dir, "forward"))
                f_state = torch.load(f_ckpt, map_location="cpu")
                self.f_model.load_state_dict(f_state["model"])

                b_ckpt = utils.best_checkpoint(os.path.join(model_dir, "backward"))
                b_state = torch.load(b_ckpt, map_location="cpu")
                self.b_model.load_state_dict(b_state["model"])
            else:
                del self.b_model.source_embedding
                del self.b_model.target_embedding
                self.b_model.source_embedding = self.f_model.target_embedding
                self.b_model.target_embedding = self.f_model.source_embedding

    def forward(self, features):
        inverse_features = {
            "source": features["target"],
            "target": features["source"],
            "source_mask": features["target_mask"],
            "target_mask": features["source_mask"]
        }

        f_state = {}
        f_state = self.f_model.encode(features, f_state)
        f_logits, f_state = self.f_model.decode(features, f_state)
        f_loss, f_log_output = self.f_model.cal_loss(f_logits, features["target"], features["target_mask"])
        f_weights = f_state["decoder_attn"]  # b x 1 x h x nq x nk
        f_weights_final = f_weights[:, -1, :, :, :].mean(dim=1) # b x nq x nk

        b_state = {}
        b_state = self.b_model.encode(inverse_features, b_state)
        b_logits, b_state = self.b_model.decode(inverse_features, b_state)
        b_loss, b_log_output = self.b_model.cal_loss(b_logits, inverse_features["target"], inverse_features["target_mask"])
        b_weights = b_state["decoder_attn"]  # b x 1 x h x nk x nq
        b_weights_final = b_weights[:, -1, :, :, :].mean(dim=1) # b x nk x nq

        bidirection_mask = self.bidir_mask(features["source_mask"], features["target_mask"]) # b x nq x nk
        weights_diff = (f_weights_final - b_weights_final.transpose(-1,-2))**2 # b x nq x nk
        agree_loss = torch.sum(weights_diff * bidirection_mask) / torch.sum(bidirection_mask)

        alpha = getattr(self.params, "agree_alpha", 5)
        loss = (f_loss + b_loss) * 0.5 + alpha * agree_loss

        log_output = "loss: {:.3f}, f_loss: {:.3f}, b_loss: {:.3f}, agree_loss: {:.3f}".format(loss, f_loss, b_loss, alpha * agree_loss)


        if getattr(self.params, "entropy_loss", False):
            beta = getattr(self.params, "entropy_beta", 1)
            lamb = getattr(self.params, "renorm_lamb", 0.05)

            # entropy loss
            def entropy(x, mask=None):
                logits = x * x.log()
                if mask is not None:
                    logits = logits * mask
                return -logits.sum() / mask.sum()
            
            def renormalize(weight, lamb=0.1, mask=None):
                weight_sum = weight.sum(dim=-1, keepdim=True) + 1e-9 # b x nq x 1
                if mask is not None:
                    weight_sum = weight_sum + lamb * mask.sum(dim=-1, keepdim=True)
                return (weight+lamb) * (1.0 / weight_sum)

            f_weights_renorm = renormalize(f_weights_final, lamb=lamb, mask=bidirection_mask)
            b_weights_renorm = renormalize(b_weights_final, lamb=lamb, mask=bidirection_mask.transpose(-1,-2))
            entropy_loss = (entropy(f_weights_renorm, mask=bidirection_mask) + entropy(b_weights_renorm, mask=bidirection_mask.transpose(-1,-2)))

            loss = loss + beta * entropy_loss
            log_output = "loss: {:.3f}, f_loss: {:.3f}, b_loss: {:.3f}, agree_loss: {:.3f}, entropy_loss: {:.3f}".format(loss, f_loss, b_loss, alpha * agree_loss, beta * entropy_loss)
        

        return loss, log_output
    
    def cal_alignment(self, features):
        inverse_features = {
            "source": features["target"],
            "target": features["source"],
            "source_mask": features["target_mask"],
            "target_mask": features["source_mask"]
        }
        
        src_seq = features["source"] # b x nk
        tgt_seq = features["target"] # b x nq
        src_mask = features["source_mask"] # b x nq
        tgt_mask = features["target_mask"] # b x nq

        f_state = {}
        f_state = self.f_model.encode(features, f_state)
        f_logits, f_state = self.f_model.decode(features, f_state) # b*nq x V
        f_pred = f_logits.argmax(-1).view(tgt_seq.shape[0], -1) # b x n
        f_acc_cnt = ((tgt_seq == f_pred) * tgt_mask).sum()

        b_state = {}
        b_state = self.b_model.encode(inverse_features, b_state)
        b_logits, b_state = self.b_model.decode(inverse_features, b_state) # b*nq x V

        b_pred = b_logits.argmax(-1).view(src_seq.shape[0], -1) # b x n
        b_acc_cnt = ((src_seq == b_pred) * src_mask).sum()

        acc_cnt = f_acc_cnt + b_acc_cnt
        all_cnt = tgt_mask.sum() + src_mask.sum()

        state = {}
        state["f_cross_attn"] = f_state["decoder_attn"]
        state["b_cross_attn"] = b_state["decoder_attn"].transpose(-1,-2)
        state["decoder_attn"] = state["f_cross_attn"] + state["b_cross_attn"] # b x 1 x h x nq x nk

        weights_f = state["f_cross_attn"][:, -1].mean(1)
        weights_b = state["b_cross_attn"][:, -1].mean(1)
        state["alignment_score"] = 2 * (weights_f * weights_b) / (weights_f + weights_b)

        if getattr(self.params, 'require_pred', False):
            f_pred = f_logits.argmax(-1).view(tgt_seq.shape[0], -1) # b x n
            state['pred'] = f_pred

        return acc_cnt, all_cnt, state

    @staticmethod
    def bidir_mask(src_mask, tgt_mask):
        return src_mask.unsqueeze(1) * tgt_mask.unsqueeze(-1) # b x nq x nk

    @classmethod
    def build_model(cls, params, model_cls):
        return cls(params, model_cls)
