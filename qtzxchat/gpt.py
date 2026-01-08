"""
gpt model

notable features: (done vs unfinished)
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qtzxchat.common import get_dist_info, print0
from qtzxchat.muon import Muon, DistMuon
from qtzxchat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # num query heads
    n_kv_head: int = 6 # num key/value heads (for GQA)
    n_embd: int = 768

def norm(x):
    # rmsnorm w/no learnable params
    return F.rms_norm(x, (x.size(-1),))
                      
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.cos_cached = None
        self.sin_cached = None
        self.seq_len_cached = 0

    def forward(self, seq_len, device):
        if seq_len > self.seq_len_cached or self.inv_freq is None or self.inv_freq.device != device:
            self.seq_len_cached = seq_len
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            self.inv_freq = inv_freq
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, inv_freq)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # relu^2 activation
        x = self.c_proj(x)
        return x








