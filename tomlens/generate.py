"""Procedural Theory-of-Mind story generator with minimal-pair variants.

Every story is built from an abstract event skeleton, so the ground truth
comes from the simulator in `world.py`, never from a template's say-so.

Skeletons (X is the agent who leaves the room, Y is the one who acts):

  classic      X puts obj in c1 / X leaves / SLOT / Y moves obj c1 -> c2
  double_move  X puts obj in c1 / Y moves c1 -> c2 / X leaves / SLOT / Y moves c2 -> c1
  swapped      Y puts obj in c1 / X leaves / SLOT / Y moves c1 -> c2

SLOT is the one sentence that differs between the three *variants* of a story:

  away    X goes somewhere else             X misses the move      (false belief)
  return  X comes back into the room        everyone sees X see it (true belief)
  watch   X secretly watches through a window: X knows, but Y thinks
          X does not -- a second-order false belief with a true first-order one

The three variants share every other sentence, word for word. That is what
lets the triplet analysis ask whether a representation groups stories by who
knows what, or by the words on the page.

The double_move skeleton puts the false belief on the *second* container
mentioned, so "the answer is the first container" is not a shortcut, and the
`return` variant contains "leaves" too, so the word "leaves" is not one.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .world import Event, simulate, answer

NAMES = ["Sally", "Anne", "Max", "Mia", "Leo", "Zoe", "Omar", "Lily", "Ivan", "Nora",
         "Raj", "Emma", "Hugo", "Ava", "Kenji", "Ruth", "Tom", "Sara", "Ben", "Chloe"]
OBJECTS = ["ball", "key", "apple", "book", "coin", "ring", "pen", "toy", "phone", "cookie",
           "marble", "watch", "letter", "candle", "spoon", "card"]
CONTAINERS = ["basket", "box", "drawer", "cupboard", "bag", "suitcase", "jar", "crate",
              "chest", "bucket", "envelope", "backpack", "basin", "tin"]
ROOMS = ["kitchen", "garage", "classroom", "bedroom", "office", "attic", "library", "studio"]
ELSEWHERE = ["the garden", "the shop", "the park", "school", "the bakery", "the post office"]

SKELETONS = ("classic", "double_move", "swapped")
VARIANTS = ("away", "return", "watch")

T_INTRO = ["{X} and {Y} are in the {room}. There is a {ca} and a {cb} in the room.",
           "{Y} and {X} are in the {room}, which has a {ca} and a {cb}.",
           "In the {room} there is a {ca} and a {cb}. {X} and {Y} are there."]
T_PUT = ["{P} puts the {obj} in the {c}.", "{P} places the {obj} inside the {c}.",
         "{P} drops the {obj} into the {c}."]
T_MOVE = ["{P} takes the {obj} out of the {c0} and puts it in the {c}.",
          "{P} moves the {obj} from the {c0} to the {c}.",
          "{P} transfers the {obj} from the {c0} into the {c}."]
T_EXIT = ["{P} leaves the {room}.", "{P} walks out of the {room}.", "{P} steps outside."]
T_SLOT = {
    "away":   ["{P} goes to {elsewhere}.", "{P} spends an hour at {elsewhere}."],
    "return": ["{P} comes back into the {room}.", "{P} walks back into the {room}."],
    "watch":  ["{P} secretly watches through the window.",
               "{P} peeks in through the window without being seen."],
}
T_FILLER = ["The sun is shining outside.", "A dog barks somewhere down the street.",
            "{Y} hums a song.", "The clock on the wall ticks.", "It starts to rain.",
            "{X} thinks about dinner.", "A bird lands on the windowsill.",
            "The radio plays quietly."]

Q_TEXT = {
    0: ("Where is the {obj} really?", "The {obj} is in the"),
    1: ("Where does {T} think the {obj} is?", "{T} thinks the {obj} is in the"),
    2: ("Where does {T} think {A} thinks the {obj} is?", "{T} thinks {A} thinks the {obj} is in the"),
}


@dataclass
class Question:
    qid: str
    qtype: str             # reality | belief_leaver | belief_actor | nested_actor_about_leaver | nested_leaver_about_actor
    order: int
    prompt: str            # story + question, ending right before the answer word
    answer: str            # container name
    foil: str              # the other container
    reality: str
    false_belief: bool     # answer differs from reality


@dataclass
class Story:
    sid: str
    group: str             # shared by the three variants of one skeleton draw
    skeleton: str
    variant: str
    split: str
    text: str
    sentences: List[str]
    slot_index: int        # index of the sentence that differs across variants
    names: Dict[str, str]
    obj: str
    containers: List[str]
    questions: List[Question] = field(default_factory=list)


def _events(skeleton, variant, X, Y, c1, c2):
    """Abstract event list (with the slot) for one skeleton/variant."""
    slot = {"away": [], "return": [Event("enter", X)], "watch": [Event("watch", X)]}[variant]
    if skeleton == "classic":
        pre = [Event("put", X, c1), Event("exit", X)]
        post = [Event("move", Y, c2)]
    elif skeleton == "double_move":
        pre = [Event("put", X, c1), Event("move", Y, c2), Event("exit", X)]
        post = [Event("move", Y, c1)]
    elif skeleton == "swapped":
        pre = [Event("put", Y, c1), Event("exit", X)]
        post = [Event("move", Y, c2)]
    else:
        raise ValueError(skeleton)
    return pre, slot, post


class _Draw:
    """One random surface realisation shared by the three variants."""

    def __init__(self, rng: random.Random, names, objects, containers, n_fill):
        self.X, self.Y = rng.sample(names, 2)
        self.obj = rng.choice(objects)
        self.c1, self.c2 = rng.sample(containers, 2)
        self.room = rng.choice(ROOMS)
        self.elsewhere = rng.choice(ELSEWHERE)
        self.intro = rng.choice(T_INTRO)
        self.mention = [self.c1, self.c2] if rng.random() < .5 else [self.c2, self.c1]
        self.t_put = rng.choice(T_PUT)
        self.t_move = [rng.choice(T_MOVE), rng.choice(T_MOVE)]
        self.t_exit = rng.choice(T_EXIT)
        self.t_slot = {v: rng.choice(T_SLOT[v]) for v in VARIANTS}
        self.fillers = rng.sample(T_FILLER, n_fill)
        self.filler_pos = sorted(rng.randint(1, 4) for _ in range(n_fill))


def _realise(d: _Draw, skeleton: str, variant: str):
    """Render sentences; return (sentences, slot_index, events)."""
    X, Y, c1, c2 = d.X, d.Y, d.c1, d.c2
    f = dict(X=X, Y=Y, obj=d.obj, room=d.room, elsewhere=d.elsewhere,
             ca=d.mention[0], cb=d.mention[1])
    pre, slot, post = _events(skeleton, variant, X, Y, c1, c2)
    sents, kinds = [d.intro.format(**f)], ["intro"]
    move_i = 0
    cur = None
    for ev in pre + [None] + post:
        if ev is None:                                         # the slot sentence
            sents.append(d.t_slot[variant].format(P=X, **{k: v for k, v in f.items()}))
            kinds.append("slot")
            continue
        if ev.kind == "put":
            sents.append(d.t_put.format(P=ev.agent, obj=d.obj, c=ev.container)); cur = ev.container
        elif ev.kind == "move":
            sents.append(d.t_move[move_i].format(P=ev.agent, obj=d.obj, c0=cur, c=ev.container))
            move_i += 1; cur = ev.container
        elif ev.kind == "exit":
            sents.append(d.t_exit.format(P=ev.agent, room=d.room))
        kinds.append(ev.kind)
    # Fillers go in at fixed positions shared by all three variants.
    for k, (txt, pos) in enumerate(zip(d.fillers, d.filler_pos)):
        sents.insert(pos + k, txt.format(X=X, Y=Y))
        kinds.insert(pos + k, "filler")
    return sents, kinds.index("slot"), pre + slot + post


def make_story(d: _Draw, skeleton, variant, sid, group, split) -> Story:
    sents, slot_i, events = _realise(d, skeleton, variant)
    X, Y = d.X, d.Y
    st = simulate([X, Y], events, present=[X, Y])
    text = " ".join(sents)
    story = Story(sid, group, skeleton, variant, split, text, sents, slot_i,
                  {"leaver": X, "actor": Y}, d.obj, [d.c1, d.c2])
    specs = [
        ("reality", 0, None, None),
        ("belief_leaver", 1, X, None),
        ("belief_actor", 1, Y, None),
        ("nested_actor_about_leaver", 2, Y, X),
        ("nested_leaver_about_actor", 2, X, Y),
    ]
    for qtype, order, T, A in specs:
        ans = answer(st, order, T, A)
        q, lead = Q_TEXT[order]
        q = q.format(obj=d.obj, T=T, A=A)
        lead = lead.format(obj=d.obj, T=T, A=A)
        prompt = f"{text}\nQuestion: {q}\nAnswer: {lead}"
        foil = d.c2 if ans == d.c1 else d.c1
        story.questions.append(Question(f"{sid}:{qtype}", qtype, order, prompt, ans, foil,
                                        st.reality, ans != st.reality))
    return story


def generate(n_groups=400, seed=0, test_frac=0.25, max_fillers=2, split_name=None) -> List[Story]:
    """n_groups skeleton draws x 3 variants = 3 * n_groups stories.

    The test split holds out *combinations*: a (leaver, object, container
    pair) never seen together in training, so a model cannot pass by
    memorising a story. Every individual word is seen in training.
    """
    rng = random.Random(seed)
    stories, seen_train = [], set()
    for g in range(n_groups):
        skeleton = SKELETONS[g % len(SKELETONS)]
        d = _Draw(rng, NAMES, OBJECTS, CONTAINERS, rng.randint(0, max_fillers))
        key = (d.X, d.obj, frozenset((d.c1, d.c2)))
        if split_name:
            split = split_name
        else:
            split = "test" if (rng.random() < test_frac and key not in seen_train) else "train"
            if split == "train":
                seen_train.add(key)
        for variant in VARIANTS:
            sid = f"g{g:04d}-{variant}"
            stories.append(make_story(d, skeleton, variant, sid, f"g{g:04d}", split))
    if not split_name:
        # A test combination drawn before an identical train one would leak; drop it.
        stories = [s for s in stories if s.split == "train"
                   or (s.names["leaver"], s.obj, frozenset(s.containers)) not in seen_train]
    return stories


def questions(stories, split=None, qtype=None):
    for s in stories:
        if split and s.split != split:
            continue
        for q in s.questions:
            if qtype and q.qtype != qtype:
                continue
            yield s, q


def save_jsonl(stories, path):
    with open(path, "w") as f:
        for s in stories:
            f.write(json.dumps(asdict(s)) + "\n")


def load_jsonl(path) -> List[Story]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d["questions"] = [Question(**q) for q in d["questions"]]
            out.append(Story(**d))
    return out
