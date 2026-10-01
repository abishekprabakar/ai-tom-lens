"""One interface over two kinds of model: the from-scratch TinyLM checkpoints and
any Hugging Face causal LM (Qwen3, DeepSeek-R1-Distill, Llama, SmolLM, ...).

For each prompt a backend returns

  hidden     (L+1, d)  residual stream at the last prompt token, every layer
  margin     logprob(answer) - logprob(foil), each scored as the full
             continuation " <container>" (so multi-token names are handled)
  lens       (L+1, 2)  logit-lens logits of the first token of answer / foil
             at every layer (final norm + unembedding applied early)

and, for a handful of showcase prompts, the full token-level view the web
tool draws: per-layer top-k next-token logits and per-token surprisal.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class Backend:
    name: str
    n_layers: int

    def encode(self, text: str) -> List[int]: ...
    def forward(self, ids: List[int]): ...          # -> (logits[T,V], hidden list of [T,d])
    def lens(self, h: torch.Tensor, layer: int) -> torch.Tensor: ...
    def token_str(self, i: int) -> str: ...
    def continuation_ids(self, prompt: str, word: str) -> List[int]: ...

    # -------------------------------------------------------------- shared --
    @torch.no_grad()
    def score(self, prompt: str, answer: str, foil: str):
        ids = self.encode(prompt)
        logits, hidden = self.forward(ids)
        last = len(ids) - 1
        H = torch.stack([h[last] for h in hidden]).float()          # (L+1, d)
        a_ids, f_ids = self.continuation_ids(prompt, answer), self.continuation_ids(prompt, foil)
        lp_a = self._cont_logprob(ids, a_ids, logits)
        lp_f = self._cont_logprob(ids, f_ids, logits)
        lens = torch.stack([self.lens(H[l], l)[[a_ids[0], f_ids[0]]] for l in range(len(hidden))])
        return H.numpy(), lp_a - lp_f, lens.float().numpy()

    def _cont_logprob(self, ids, cont, logits_prompt):
        lp = F.log_softmax(logits_prompt[-1].float(), -1)[cont[0]].item()
        if len(cont) > 1:                       # teacher-force the rest of the word
            full = ids + cont
            logits, _ = self.forward(full)
            for k in range(1, len(cont)):
                lp += F.log_softmax(logits[len(ids) + k - 1].float(), -1)[cont[k]].item()
        return lp

    @torch.no_grad()
    def token_view(self, prompt: str, topk: int = 50, track=()):
        """Per-layer top-k next-token logits at the last position + per-token
        surprisal. Token ids in `track` (answer / foil) are always recorded,
        whether or not they make the top k."""
        ids = self.encode(prompt)
        logits, hidden = self.forward(ids)
        last = len(ids) - 1
        layers = []
        for l, h in enumerate(hidden):
            lg = self.lens(h[last].float(), l).float()
            k = min(topk, lg.shape[-1])
            v, i = lg.topk(k)
            layers.append({"ids": i.tolist(), "logits": [round(x, 4) for x in v.tolist()],
                           "tokens": [self.token_str(t) for t in i.tolist()],
                           "tracked": [round(lg[t].item(), 4) for t in track],
                           "logsumexp": round(torch.logsumexp(lg, -1).item(), 4)})
        lsm = F.log_softmax(logits.float(), -1)
        surprisal = [None] + [round(-lsm[t - 1, ids[t]].item() / math.log(2), 3) for t in range(1, len(ids))]
        return {"tokens": [self.token_str(t) for t in ids], "surprisal_bits": surprisal,
                "layers": layers, "vocab_size": int(logits.shape[-1])}


class TinyBackend(Backend):
    def __init__(self, path, name=None):
        from .tinylm import load_model
        self.model, self.tok = load_model(path)
        self.name = name or self.model.arch
        self.n_layers = self.model.n_layers

    def encode(self, text):
        return self.tok.encode(text)

    def forward(self, ids):
        logits, hidden = self.model(torch.tensor([ids]))
        return logits[0], [h[0] for h in hidden]

    def lens(self, h, layer):
        return self.model.lens(h)

    def token_str(self, i):
        return self.tok.itos[i]

    def continuation_ids(self, prompt, word):
        return [self.tok.stoi[word.lower()]]


class HFBackend(Backend):
    """Any transformers causal LM whose decoder exposes `.model.norm` and
    `.lm_head` (Llama, Qwen2, Qwen3, Mistral, Gemma, OLMo, SmolLM, ...).

    Note on the logit lens: in these implementations the *last* entry of
    `hidden_states` already has the final norm applied, so it goes straight
    to lm_head; every earlier layer gets norm + lm_head.
    """

    def __init__(self, model_id, model=None, tokenizer=None, device=None, dtype=None, name=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        if model is None:
            dtype = dtype or (torch.float32 if device == "cpu" else torch.float16)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model = model.to(device).eval()
        self.name = name or model_id.split("/")[-1]
        self.n_layers = self.model.config.num_hidden_layers
        base = getattr(self.model, "model", None)
        self._norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
        if self._norm is None:
            raise ValueError("model has no final norm at .model.norm; logit lens unsupported")

    def encode(self, text):
        ids = self.tok(text, add_special_tokens=True)["input_ids"]
        return list(ids)

    def forward(self, ids):
        out = self.model(torch.tensor([ids], device=self.device), output_hidden_states=True)
        return out.logits[0].cpu(), [h[0].cpu() for h in out.hidden_states]

    def lens(self, h, layer):
        h = h.to(self.device, dtype=self.model.lm_head.weight.dtype)
        if layer < self.n_layers:
            h = self._norm(h)
        return self.model.lm_head(h).float().cpu()

    def token_str(self, i):
        return self.tok.decode([i])

    def continuation_ids(self, prompt, word):
        """Tokens that " word" adds after the prompt (handles merges at the seam)."""
        base = self.encode(prompt)
        full = self.encode(prompt + " " + word)
        k = 0
        while k < len(base) and k < len(full) and base[k] == full[k]:
            k += 1
        if k < len(base):
            raise ValueError(f"tokenisation of the prompt changed when {word!r} was appended")
        return full[k:]
