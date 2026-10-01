"""Three small language-model architectures, trained from scratch on the corpus.

They stand in for pretrained LLMs where weights can't be downloaded, and they
make a useful contrast on their own:

  gpt     pre-LN transformer, learned absolute positions, GELU MLP  (GPT-2 family)
  llama   pre-RMSNorm transformer, rotary positions, SwiGLU MLP     (Llama / Qwen family)
  gru     stacked GRU, no attention at all                          (recurrent baseline)

Every model exposes the same interface the analysis code uses:

  forward(ids) -> (logits, hidden)   hidden[l] is the residual stream after
                                     layer l (hidden[0] = embeddings), shape (B, T, d)
  lens(h)      -> logits             final norm + unembedding applied to any
                                     hidden state: the "logit lens"

so lens(hidden[-1]) == logits exactly, which the tests check.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_TOK = re.compile(r"\n|[A-Za-z']+|[.,?:!]")


class WordTokenizer:
    """Lower-cased word-level tokenizer. The corpus vocabulary is closed, so
    a word tokenizer keeps every container name a single token."""

    def __init__(self, vocab: Sequence[str]):
        self.itos = ["<pad>", "<unk>", "<bos>"] + sorted(set(vocab) - {"<pad>", "<unk>", "<bos>"})
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.unk, self.bos = 0, 1, 2

    @staticmethod
    def split(text: str) -> List[str]:
        return [t.lower() for t in _TOK.findall(text)]

    @classmethod
    def fit(cls, texts):
        vocab = set()
        for t in texts:
            vocab.update(cls.split(t))
        return cls(vocab)

    def encode(self, text: str, bos=True) -> List[int]:
        ids = [self.stoi.get(w, self.unk) for w in self.split(text)]
        return ([self.bos] if bos else []) + ids

    def decode(self, ids) -> str:
        return " ".join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.itos)


# ------------------------------------------------------------------ blocks --

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, T, Dh), rotate-half convention."""
    B, H, T, D = x.shape
    half = D // 2
    inv = base ** (-torch.arange(half, dtype=x.dtype, device=x.device) / half)
    ang = torch.arange(T, dtype=x.dtype, device=x.device)[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, d, heads, use_rope, bias):
        super().__init__()
        self.h, self.dh, self.use_rope = heads, d // heads, use_rope
        self.qkv = nn.Linear(d, 3 * d, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)
        self.last_attn = None

    def forward(self, x, keep_attn=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        if keep_attn:
            self.last_attn = att.detach()
        return self.o((att @ v).transpose(1, 2).reshape(B, T, d))


class GPTBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, use_rope=False, bias=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, keep_attn=False):
        x = x + self.attn(self.n1(x), keep_attn)
        return x + self.mlp(self.n2(x))


class LlamaBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, heads, use_rope=True, bias=False)
        hid = int(8 * d / 3)
        self.gate = nn.Linear(d, hid, bias=False)
        self.up = nn.Linear(d, hid, bias=False)
        self.down = nn.Linear(hid, d, bias=False)

    def forward(self, x, keep_attn=False):
        x = x + self.attn(self.n1(x), keep_attn)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


# ------------------------------------------------------------------ models --

class TinyLM(nn.Module):
    def __init__(self, arch: str, vocab: int, d=128, layers=4, heads=4, max_len=320, tie=False):
        super().__init__()
        self.arch, self.d, self.n_layers = arch, d, layers
        self.config = dict(arch=arch, vocab=vocab, d=d, layers=layers, heads=heads, max_len=max_len, tie=tie)
        self.emb = nn.Embedding(vocab, d)
        if arch == "gpt":
            self.pos = nn.Embedding(max_len, d)
            self.blocks = nn.ModuleList(GPTBlock(d, heads) for _ in range(layers))
            self.norm = nn.LayerNorm(d)
        elif arch == "llama":
            self.blocks = nn.ModuleList(LlamaBlock(d, heads) for _ in range(layers))
            self.norm = RMSNorm(d)
        elif arch == "gru":
            self.blocks = nn.ModuleList(nn.GRU(d, d, batch_first=True) for _ in range(layers))
            self.norm = nn.LayerNorm(d)
        else:
            raise ValueError(arch)
        self.head = nn.Linear(d, vocab, bias=False)
        if tie:
            # Off by default. Tied embeddings let the language-modelling loss
            # pull every name toward one shared vector (all names are equally
            # likely after "."), and the model can then no longer match the
            # name in a question to the same name in the story.
            self.head.weight = self.emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, keep_attn=False):
        x = self.emb(ids)
        if self.arch == "gpt":
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        hidden = [x]
        for blk in self.blocks:
            if self.arch == "gru":
                x = x + blk(x)[0]                    # residual stack, like the transformers
            else:
                x = blk(x, keep_attn)
            hidden.append(x)
        return self.lens(x), hidden

    def lens(self, h):
        return self.head(self.norm(h))


def save_model(model: TinyLM, tok: WordTokenizer, path):
    torch.save({"config": model.config, "state": model.state_dict(), "vocab": tok.itos}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TinyLM(**ck["config"])
    m.load_state_dict(ck["state"])
    m.eval()
    tok = WordTokenizer([])
    tok.itos = ck["vocab"]
    tok.stoi = {w: i for i, w in enumerate(tok.itos)}
    return m, tok
