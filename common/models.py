#!/usr/bin/env python3
"""Model definitions for the two SOTA targets.

Target 1 (Abdelmajeed et al., CMC 83(2) 2025, DOI 10.32604/cmc.2025.059870):
  BERT backbone + deep label predictor (CE) + binary domain classifier (BCE)
  connected through a Gradient Reversal Layer with progressive schedule
  lambda_p = 2/(1+exp(-gamma*P)) - 1 (Eq. 11). Total loss = label CE +
  lambda_adv * domain loss, lambda_adv = 1e-2 (paper section 5.2).

Target 2 (Abdul-Mageed et al., ACL 2021, ARBERT & MARBERT):
  Plain BERT backbone + classification head (CE). Unlabeled benefit is in
  pretraining (1B tweets); fine-tuning is supervised.
"""
import math

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class GradReverse(torch.autograd.Function):
    """Identity in forward; multiplies gradient by -lambda in backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def progressive_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """lambda_p = 2/(1+exp(-gamma*P)) - 1, P in [0,1] (paper Eq. 11).

    gamma=10 is the canonical DANN value (Ganin & Lempitsky 2015, ref [49] in
    the paper); the paper does not state gamma explicitly — recorded assumption.
    """
    if total_steps <= 1:
        p = 1.0
    else:
        p = min(1.0, max(0.0, step / total_steps))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class GRLModel(nn.Module):
    """Target 1: backbone + label predictor + domain classifier via GRL."""

    def __init__(self, backbone_name: str, num_labels: int, dropout: float = 0.5):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone_name)
        cfg.hidden_dropout_prob = dropout
        cfg.classifier_dropout = getattr(cfg, "classifier_dropout", dropout)
        self.bert = AutoModel.from_pretrained(backbone_name, config=cfg)
        hidden = self.bert.config.hidden_size
        self.label_head = nn.Linear(hidden, num_labels)      # lr 5e-4
        self.domain_head = nn.Linear(hidden, 1)               # lr 1e-4
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, grl_lambda: float):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]          # [CLS] representation
        logits_label = self.label_head(h)           # (B, num_labels)
        h_rev = GradReverse.apply(h, grl_lambda)
        logits_domain = self.domain_head(h_rev).squeeze(-1)   # (B,)
        return logits_label, logits_domain


class PlainClassifier(nn.Module):
    """Target 2: backbone + classification head (supervised fine-tuning)."""

    def __init__(self, backbone_name: str, num_labels: int, dropout: float = 0.5):
        super().__init__()
        cfg = AutoConfig.from_pretrained(backbone_name)
        cfg.hidden_dropout_prob = dropout
        cfg.classifier_dropout = getattr(cfg, "classifier_dropout", dropout)
        self.bert = AutoModel.from_pretrained(backbone_name, config=cfg)
        hidden = self.bert.config.hidden_size
        self.head = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]
        return self.head(h)


def param_groups_grl(model: GRLModel, lr_backbone: float, lr_label: float, lr_domain: float):
    backbone_params = list(model.bert.parameters())
    label_params = list(model.label_head.parameters())
    domain_params = list(model.domain_head.parameters())
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": label_params, "lr": lr_label},
        {"params": domain_params, "lr": lr_domain},
    ]


def param_groups_plain(model: PlainClassifier, lr: float):
    return [{"params": list(model.parameters()), "lr": lr}]
