"""Train the three from-scratch architectures on the training split.

Objective: ordinary next-token cross-entropy over the whole sequence
(story + question + answer), plus an extra weight on the answer token.
Without that weight the loss is dominated by the template words, which are
easy, and the one token that requires belief tracking gets 1% of the
gradient.

    python -m tomlens.train --arch gpt llama gru --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .generate import generate, questions
from .tinylm import TinyLM, WordTokenizer, save_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_corpus(seed=0):
    """The evaluation benchmark: 400 groups x 3 variants (train/test split by
    held-out combinations) plus 60 groups of longer out-of-distribution
    stories with up to four distractor sentences."""
    stories = generate(n_groups=400, seed=seed)
    ood = generate(n_groups=60, seed=seed + 7, max_fillers=4, split_name="ood")
    return stories, ood


def _key(s):
    return (s.names["leaver"], s.obj, frozenset(s.containers))


def build_training_corpus(eval_stories, n_groups=4000, seed=1000):
    """A separate, larger pretraining corpus for the from-scratch models.

    The 891 benchmark training stories are too few: a 1M-parameter model
    memorises them (99.8% train accuracy) and sits at chance on held-out
    first-order false beliefs. This corpus is drawn with a different seed and
    excludes every (leaver, object, container pair) combination that appears
    in the benchmark's test or OOD splits, so held-out evaluation stays
    held out.
    """
    banned = {_key(s) for s in eval_stories if s.split in ("test", "ood")}
    return [s for s in generate(n_groups=n_groups, seed=seed, split_name="train") if _key(s) not in banned]


def encode_examples(tok, stories, split):
    out = []
    for s, q in questions(stories, split=split):
        ids = tok.encode(q.prompt + " " + q.answer)
        out.append((ids, len(ids) - 1, q))         # position of the answer token
    return out


AUX_VOCAB = "Question: Who left the room? Who moved the? Did see where the went? Answer: yes no ."


def aux_questions(s):
    """Precursor questions used only in training (never in the benchmark).

    Each one is a single hop the belief questions need: who left, who acted,
    and "seeing leads to knowing" for a named person. They exist because
    without them none of the three architectures learned to bind the name in
    a question to that person's role (see README, "What it took to train").
    """
    X, Y = s.names["leaver"], s.names["actor"]
    saw_x = "no" if s.variant == "away" else "yes"
    return [(f"\nQuestion: Who left the room?\nAnswer:", X),
            (f"\nQuestion: Who moved the {s.obj}?\nAnswer:", Y),
            (f"\nQuestion: Did {X} see where the {s.obj} went?\nAnswer:", saw_x),
            (f"\nQuestion: Did {Y} see where the {s.obj} went?\nAnswer:", "yes")]


def encode_packed(tok, stories, rng, aux=False):
    """One training sequence per story with its questions appended in a
    random order: story / Q / A / Q / A ... Contrasting questions about the
    same story in one context is a stronger signal for binding a name to a
    role, and it is ~2.5x cheaper than separate sequences. Answer positions
    are returned so the extra answer-token weight applies to every answer."""
    out = []
    for s in stories:
        qa = [(q.prompt[len(s.text):], q.answer) for q in s.questions]
        if aux:
            qa += aux_questions(s)
        rng.shuffle(qa)
        text = s.text
        for q, a in qa:
            text += q + " " + a + " ."
        ids = tok.encode(text)
        ans_words = {a.lower() for _, a in qa}
        pos, in_answer = [], False
        for i, t in enumerate(ids):
            w = tok.itos[t]
            if w == "answer":
                in_answer = True
            elif in_answer and w == "." and tok.itos[ids[i - 1]] in ans_words:
                pos.append(i - 1)
                in_answer = False
        assert len(pos) == len(qa), (len(pos), len(qa))
        out.append((ids, pos, None))
    return out


def batches(examples, bs, shuffle, rng):
    idx = list(range(len(examples)))
    if shuffle:
        # shuffle, sort within windows of 32 batches to cut padding,
        # then shuffle the batch order again
        rng.shuffle(idx)
        win = bs * 32
        idx = [j for w in range(0, len(idx), win)
               for j in sorted(idx[w:w + win], key=lambda j: len(examples[j][0]))]
        order = list(range(0, len(idx), bs))
        rng.shuffle(order)
    else:
        order = list(range(0, len(idx), bs))
    for i in order:
        chunk = [examples[j] for j in idx[i:i + bs]]
        T = max(len(e[0]) for e in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        for r, (seq, _, _) in enumerate(chunk):
            ids[r, :len(seq)] = torch.tensor(seq)
        if isinstance(chunk[0][1], list):          # packed: boolean answer mask
            apos = torch.zeros(len(chunk), T, dtype=torch.bool)
            for r, e in enumerate(chunk):
                apos[r, e[1]] = True
            yield ids, apos, chunk
        else:
            yield ids, torch.tensor([e[1] for e in chunk]), chunk


@torch.no_grad()
def forced_choice_accuracy(model, tok, examples, bs=128):
    """Answer correct if logit(answer) > logit(foil) at the answer position."""
    model.eval()
    hits, by_type = 0, {}
    for ids, apos, chunk in batches(examples, bs, False, None):
        logits, _ = model(ids)
        rows = torch.arange(len(chunk))
        lg = logits[rows, apos - 1]
        for r, (_, _, q) in enumerate(chunk):
            ok = lg[r, tok.stoi[q.answer]] > lg[r, tok.stoi[q.foil]]
            hits += int(ok)
            key = f"{q.qtype}|{'FB' if q.false_belief else 'TB'}"
            a, n = by_type.get(key, (0, 0))
            by_type[key] = (a + int(ok), n + 1)
    model.train()
    return hits / len(examples), {k: a / n for k, (a, n) in sorted(by_type.items())}


def train(arch, tok, train_ex, test_ex, epochs=40, d=128, layers=4, heads=4, lr=2e-3,
          bs=64, answer_weight=4.0, seed=0, log=print):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyLM(arch, len(tok), d=d, layers=layers, heads=heads)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_ex) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for ids, apos, _ in batches(train_ex, bs, True, rng):
            logits, _ = model(ids)
            tgt = ids[:, 1:].clone()
            tgt[tgt == tok.pad] = -100
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                 ignore_index=-100)
            m = apos[:, 1:]                          # answer tokens, as next-token targets
            ans = F.cross_entropy(logits[:, :-1][m], ids[:, 1:][m])
            loss = ce + answer_weight * ans
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * ids.shape[0]
            n += ids.shape[0]
        if True:
            acc, bt = forced_choice_accuracy(model, tok, test_ex)
            fb1 = bt.get("belief_leaver|FB", float("nan"))
            history.append({"epoch": ep + 1, "loss": tot / n, "test_acc": acc, "test_leaver_false_belief": fb1})
            log(f"  [{arch}] epoch {ep + 1:3d}  loss {tot / n:.4f}  test acc {acc:.3f}  "
                f"leaver false belief {fb1:.3f}  ({time.time() - t0:.0f}s)")
    return model, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["gpt", "llama", "gru"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--aux", action="store_true",
                    help="also train on the precursor questions (did not help; see README)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-groups", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    stories, ood = build_corpus()
    tok = WordTokenizer.fit([q.prompt + " " + q.answer for _, q in questions(stories + ood)] + [AUX_VOCAB])
    pretrain = build_training_corpus(stories + ood, n_groups=a.train_groups)
    train_ex = encode_packed(tok, pretrain, random.Random(0), aux=a.aux)
    test_ex = encode_examples(tok, stories, "test")
    print(f"vocab {len(tok)}, pretraining {len(pretrain)} stories (5 questions packed per sequence), "
          f"benchmark test {len(test_ex)} prompts")
    os.makedirs(a.out, exist_ok=True)
    logs = {}
    for arch in a.arch:
        model, hist = train(arch, tok, train_ex, test_ex, epochs=a.epochs, bs=a.bs, lr=a.lr)
        save_model(model, tok, os.path.join(a.out, f"{arch}.pt"))
        logs[arch] = {"params": sum(p.numel() for p in model.parameters()), "history": hist}
    with open(os.path.join(a.out, "training_log.json"), "w") as f:
        json.dump(logs, f, indent=1)


if __name__ == "__main__":
    main()
