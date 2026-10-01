"""Epistemic world simulator: the ground truth for every Theory-of-Mind label.

A story is a sequence of events in one room. The simulator tracks

  * reality         where the object actually is                 (0th order)
  * belief[a]       where agent a thinks it is                    (1st order)
  * nested[a][b]    where a thinks b thinks it is                 (2nd order)

using one observation rule. When an event happens, the set of agents who
see it is `present` (everyone in the room) plus any secret watchers.

  * every observer updates their own belief;
  * observer a updates nested[a][b] for every b that a *knows* saw it --
    the agents in the room are mutually visible, a secret watcher is not;
  * a secret watcher knows who was in the room, so they update their model
    of every agent present;
  * nobody else changes anything.

That rule is what makes second-order false belief possible: if Anne secretly
watches Sally move the ball, Anne knows where it is, but Sally still thinks
Anne believes the old location.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Event:
    kind: str                     # "enter" | "exit" | "put" | "move" | "watch" | "unwatch"
    agent: str
    container: Optional[str] = None

    def __post_init__(self):
        if self.kind not in {"enter", "exit", "put", "move", "watch", "unwatch"}:
            raise ValueError(f"unknown event kind {self.kind!r}")
        if self.kind in {"put", "move"} and self.container is None:
            raise ValueError(f"{self.kind} needs a container")


@dataclass
class WorldState:
    agents: Tuple[str, ...]
    present: set = field(default_factory=set)
    watching: set = field(default_factory=set)       # outside, secretly watching
    reality: Optional[str] = None
    belief: Dict[str, Optional[str]] = field(default_factory=dict)
    nested: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)

    @classmethod
    def start(cls, agents: Sequence[str], present: Sequence[str] = ()):
        agents = tuple(agents)
        return cls(
            agents=agents,
            present=set(present),
            belief={a: None for a in agents},
            nested={a: {b: None for b in agents if b != a} for a in agents},
        )


def apply(state: WorldState, ev: Event) -> WorldState:
    a = ev.agent
    if a not in state.agents:
        raise ValueError(f"unknown agent {a!r}")
    if ev.kind == "enter":
        state.present.add(a)
        state.watching.discard(a)
    elif ev.kind == "exit":
        if a not in state.present:
            raise ValueError(f"{a} cannot leave: not in the room")
        state.present.discard(a)
    elif ev.kind == "watch":
        if a in state.present:
            raise ValueError(f"{a} is in the room; cannot watch secretly")
        state.watching.add(a)
    elif ev.kind == "unwatch":
        state.watching.discard(a)
    else:  # put / move: the only events that change where the object is
        if a not in state.present:
            raise ValueError(f"{a} must be in the room to touch the object")
        if ev.kind == "move" and state.reality is None:
            raise ValueError("cannot move an object that was never placed")
        state.reality = ev.container
        room = set(state.present)
        for obs in room | state.watching:
            state.belief[obs] = ev.container
            # Who does `obs` know saw this? Everyone visibly in the room.
            for other in room:
                if other != obs:
                    state.nested[obs][other] = ev.container
    return state


def simulate(agents: Sequence[str], events: Sequence[Event], present: Sequence[str] = ()) -> WorldState:
    st = WorldState.start(agents, present)
    for ev in events:
        apply(st, ev)
    return st


def answer(state: WorldState, order: int, target: Optional[str] = None, about: Optional[str] = None) -> str:
    """Ground-truth answer to a 0th/1st/2nd order location question.

    order 0: reality; order 1: belief[target]; order 2: nested[target][about]
    ("where does target think about thinks the object is?").
    """
    if order == 0:
        out = state.reality
    elif order == 1:
        out = state.belief[target]
    elif order == 2:
        out = state.nested[target][about]
    else:
        raise ValueError("order must be 0, 1 or 2")
    if out is None:
        raise ValueError("question has no defined answer (agent never saw the object)")
    return out
