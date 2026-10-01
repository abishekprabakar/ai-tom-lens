"""Model plumbing: causality, RoPE, and the logit-lens identity on both
backends. The Hugging Face path is exercised on tiny *randomly initialised*
Llama / Qwen2 / Qwen3 configs with a local word-level tokenizer -- no
download -- which checks the same code that runs on real checkpoints."""
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from tomlens.tinylm import TinyLM, WordTokenizer, rope, save_model
from tomlens.extract import TinyBackend, HFBackend

PROMPT = "Sally puts the ball in the basket . Sally leaves . Anne moves the ball to the box .\nAnswer : it is in the"


class TestTinyLM(unittest.TestCase):
    def test_causal_for_every_arch(self):
        torch.manual_seed(0)
        for arch in ("gpt", "llama", "gru"):
            m = TinyLM(arch, 30, d=32, layers=2, heads=4).eval()
            a = torch.randint(0, 30, (1, 12))
            b = a.clone(); b[0, 8:] = (b[0, 8:] + 1) % 30
            with torch.no_grad():
                la, _ = m(a); lb, _ = m(b)
            with self.subTest(arch=arch):
                torch.testing.assert_close(la[0, :8], lb[0, :8])
                self.assertFalse(torch.allclose(la[0, 8:], lb[0, 8:]))

    def test_rope_relative_and_norm_preserving(self):
        torch.manual_seed(1)
        q = torch.randn(1, 1, 1, 8).expand(1, 1, 10, 8).contiguous()
        k = torch.randn(1, 1, 1, 8).expand(1, 1, 10, 8).contiguous()
        rq, rk = rope(q), rope(k)
        torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))
        # q_i . k_j depends only on i - j
        dots = (rq[0, 0] @ rk[0, 0].T)
        torch.testing.assert_close(dots[5, 2], dots[7, 4])
        torch.testing.assert_close(dots[3, 3], dots[9, 9])

    def test_lens_of_last_layer_is_logits(self):
        m = TinyLM("llama", 30, d=32, layers=3, heads=4).eval()
        with torch.no_grad():
            logits, hidden = m(torch.randint(0, 30, (2, 7)))
        self.assertEqual(len(hidden), 4)
        torch.testing.assert_close(m.lens(hidden[-1]), logits)

    def test_tokenizer_round_trip(self):
        tok = WordTokenizer.fit([PROMPT])
        ids = tok.encode(PROMPT)
        self.assertEqual(ids[0], tok.bos)
        self.assertEqual(tok.decode(ids[1:]), " ".join(WordTokenizer.split(PROMPT)))
        self.assertIn("\n", tok.stoi)


class TestTinyBackend(unittest.TestCase):
    def test_score_matches_manual_computation(self):
        torch.manual_seed(2)
        tok = WordTokenizer.fit([PROMPT + " basket box"])
        m = TinyLM("gpt", len(tok), d=32, layers=2, heads=4).eval()
        with tempfile.TemporaryDirectory() as d:
            save_model(m, tok, os.path.join(d, "m.pt"))
            be = TinyBackend(os.path.join(d, "m.pt"))
        H, margin, lens = be.score(PROMPT, "box", "basket")
        ids = torch.tensor([tok.encode(PROMPT)])
        with torch.no_grad():
            logits, hidden = m(ids)
        lp = F.log_softmax(logits[0, -1], -1)
        self.assertAlmostEqual(margin, (lp[tok.stoi["box"]] - lp[tok.stoi["basket"]]).item(), places=5)
        self.assertEqual(H.shape, (3, 32))
        self.assertAlmostEqual(float(lens[-1, 0]), logits[0, -1, tok.stoi["box"]].item(), places=4)
        view = be.token_view(PROMPT, topk=5, track=(tok.stoi["box"], tok.stoi["basket"]))
        self.assertEqual(len(view["layers"]), 3)
        self.assertAlmostEqual(view["layers"][-1]["tracked"][0], logits[0, -1, tok.stoi["box"]].item(), places=3)
        self.assertIsNone(view["surprisal_bits"][0])


def _word_tokenizer(words):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {w: i for i, w in enumerate(["<unk>", "<s>"] + sorted(set(words)))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", bos_token="<s>")


class TestHFBackend(unittest.TestCase):
    def _configs(self):
        from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                                  Qwen3Config, Qwen3ForCausalLM)
        kw = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128)
        return [("llama", LlamaConfig, LlamaForCausalLM, kw), ("qwen2", Qwen2Config, Qwen2ForCausalLM, kw),
                ("qwen3", Qwen3Config, Qwen3ForCausalLM, dict(kw, head_dim=8))]

    def test_logit_lens_and_scoring(self):
        words = PROMPT.split() + ["basket", "box"]
        tok = _word_tokenizer(words)
        for name, C, M, kw in self._configs():
            torch.manual_seed(3)
            model = M(C(vocab_size=len(tok), **kw)).eval()
            be = HFBackend(name, model=model, tokenizer=tok, device="cpu")
            with self.subTest(arch=name):
                ids = be.encode(PROMPT)
                logits, hidden = be.forward(ids)
                self.assertEqual(len(hidden), 4)
                # last hidden state is already normed: lens(final) must equal the logits
                torch.testing.assert_close(be.lens(hidden[-1][-1], 3), logits[-1], rtol=1e-4, atol=1e-4)
                # an intermediate layer gets the norm applied
                manual = model.lm_head(model.model.norm(hidden[1][-1]))
                torch.testing.assert_close(be.lens(hidden[1][-1], 1), manual, rtol=1e-4, atol=1e-4)
                H, margin, lens = be.score(PROMPT, "box", "basket")
                a, f = be.continuation_ids(PROMPT, "box"), be.continuation_ids(PROMPT, "basket")
                lp = F.log_softmax(logits[-1], -1)
                self.assertAlmostEqual(margin, (lp[a[0]] - lp[f[0]]).item(), places=4)
                self.assertEqual(H.shape, (4, 32))
                self.assertAlmostEqual(float(lens[-1, 0]), logits[-1, a[0]].item(), places=3)

    def test_multi_token_continuation(self):
        # "cupboard" is not in the vocabulary as a word here; build a tokenizer
        # where it is two tokens by giving it a space-free split instead.
        from transformers import LlamaConfig, LlamaForCausalLM
        tok = _word_tokenizer(PROMPT.split() + ["cup", "board", "box"])
        torch.manual_seed(4)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
        be = HFBackend("llama", model=model, tokenizer=tok, device="cpu")
        cont = be.continuation_ids(PROMPT, "cup board")
        self.assertEqual(len(cont), 2)
        ids = be.encode(PROMPT)
        lp = be._cont_logprob(ids, cont, be.forward(ids)[0])
        full_logits, _ = be.forward(ids + cont)
        ls = F.log_softmax(full_logits, -1)
        manual = ls[len(ids) - 1, cont[0]] + ls[len(ids), cont[1]]
        self.assertAlmostEqual(lp, manual.item(), places=4)


if __name__ == "__main__":
    unittest.main()
