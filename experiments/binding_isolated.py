"""The binding wall, isolated.

Trains a fresh model on ONE sub-task: in `away` stories, answer "where does
T think the object is?" where T is either the person who left (stale
container) or the person who moved it (current container). The two
prompts for a story differ only in the name T, so the task is exactly
"bind the name in the question to that person's role in the story".

For contrast it also trains the same model on "Who left the room?", which
needs the role but not a match against a name in the question.

    python experiments/binding_isolated.py --arch llama --steps 2000

Measured on a 2-core CPU: "who left" is at 100% held-out accuracy by step
250; the binding task stays at 50% (loss around ln 2) through 2,000 steps
for llama (d=128, 4 layers), gpt (d=128, 3 layers, 8 heads, lr 3e-3) and
gru (d=128, 2 layers). Output of the default run is in the README.
"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tomlens.generate import questions                         # noqa: E402
from tomlens.tinylm import TinyLM, WordTokenizer                # noqa: E402
from tomlens.train import (AUX_VOCAB, aux_questions, batches,   # noqa: E402
                           build_corpus, build_training_corpus)


def examples(tok, stories, task):
    out = []
    for st in stories:
        if task == "who_left":
            q, a = aux_questions(st)[0]
            ids = tok.encode(st.text + q + " " + a)
            out.append((ids, len(ids) - 1, None))
        elif st.variant == "away":
            for q in st.questions:
                if q.qtype in ("belief_leaver", "belief_actor"):
                    ids = tok.encode(q.prompt + " " + q.answer)
                    out.append((ids, len(ids) - 1, (tok.stoi[q.answer.lower()], tok.stoi[q.foil.lower()])))
    return out


@torch.no_grad()
def accuracy(model, data):
    hit = 0
    for ids, apos, extra in batches(data, 128, False, None):
        lg, _ = model(ids)
        lg = lg[torch.arange(len(extra)), apos - 1]
        for r, (_, _, e) in enumerate(extra):
            if e is None:                       # who_left: exact name, argmax over vocabulary
                hit += int(lg[r].argmax() == ids[r, apos[r]])
            else:                               # binding: forced choice between the containers
                hit += int(lg[r, e[0]] > lg[r, e[1]])
    return hit / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="llama")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--task", nargs="+", default=["who_left", "binding"])
    a = ap.parse_args()
    s, o = build_corpus()
    pre = build_training_corpus(s + o)
    tok = WordTokenizer.fit([q.prompt + " " + q.answer for _, q in questions(s + o)] + [AUX_VOCAB])
    test_stories = [x for x in s if x.split == "test"]
    for task in a.task:
        torch.manual_seed(0)
        rng = random.Random(0)
        train, test = examples(tok, pre, task), examples(tok, test_stories, task)
        model = TinyLM(a.arch, len(tok), d=a.d, layers=a.layers, heads=a.heads)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        step, t0 = 0, time.time()
        while step < a.steps:
            for ids, apos, _ in batches(train, 32, True, rng):
                lg, _ = model(ids)
                r = torch.arange(ids.shape[0])
                loss = F.cross_entropy(lg[r, apos - 1], ids[r, apos])
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1
                if step % 250 == 0:
                    model.eval()
                    print(f"{task:9s} {a.arch} step {step:5d}  loss {loss.item():.3f}  "
                          f"held-out acc {accuracy(model, test):.3f}  ({time.time() - t0:.0f}s)", flush=True)
                    model.train()
                if step >= a.steps:
                    break


if __name__ == "__main__":
    main()
