"""The ground truth has to be right before anything else means anything.

Expected answers here are worked out by hand from the story, not read off the
simulator.
"""
import re
import unittest
from collections import Counter

from tomlens.world import Event, simulate, answer
from tomlens.generate import generate, questions, VARIANTS


class TestWorld(unittest.TestCase):
    def test_sally_anne(self):
        # Sally puts the marble in the basket and leaves; Anne moves it to the box.
        st = simulate(["Sally", "Anne"], [Event("put", "Sally", "basket"), Event("exit", "Sally"),
                                          Event("move", "Anne", "box"), Event("enter", "Sally")],
                      present=["Sally", "Anne"])
        self.assertEqual(answer(st, 0), "box")
        self.assertEqual(answer(st, 1, "Sally"), "basket")              # first-order false belief
        self.assertEqual(answer(st, 1, "Anne"), "box")
        self.assertEqual(answer(st, 2, "Anne", "Sally"), "basket")      # Anne knows Sally missed it
        self.assertEqual(answer(st, 2, "Sally", "Anne"), "basket")      # Sally doesn't know Anne moved it

    def test_secret_watcher_makes_second_order_false_belief(self):
        st = simulate(["Sally", "Anne"], [Event("put", "Sally", "basket"), Event("exit", "Sally"),
                                          Event("watch", "Sally"), Event("move", "Anne", "box")],
                      present=["Sally", "Anne"])
        self.assertEqual(answer(st, 1, "Sally"), "box")                 # she saw it
        self.assertEqual(answer(st, 2, "Anne", "Sally"), "basket")      # Anne doesn't know that
        self.assertEqual(answer(st, 2, "Sally", "Anne"), "box")         # Sally knows Anne knows

    def test_returning_before_the_move_is_true_belief(self):
        st = simulate(["A", "B"], [Event("put", "A", "jar"), Event("exit", "A"), Event("enter", "A"),
                                   Event("move", "B", "tin")], present=["A", "B"])
        self.assertEqual(answer(st, 1, "A"), "tin")
        self.assertEqual(answer(st, 2, "B", "A"), "tin")

    def test_invalid_events(self):
        with self.assertRaises(ValueError):
            simulate(["A", "B"], [Event("exit", "A"), Event("put", "A", "jar")], present=["A", "B"])
        with self.assertRaises(ValueError):
            simulate(["A"], [Event("move", "A", "jar")], present=["A"])
        with self.assertRaises(ValueError):
            Event("teleport", "A")
        st = simulate(["A", "B"], [Event("exit", "A"), Event("put", "B", "jar")], present=["A", "B"])
        with self.assertRaises(ValueError):
            answer(st, 1, "A")                                           # A never saw the object


# Hand-derived truth table: which (skeleton, variant) make each question a
# false-belief question, and whether the false belief points at the first
# container (c1) or the second.
FB_TABLE = {
    "belief_leaver": {"away"},
    "nested_actor_about_leaver": {"away", "watch"},
    "nested_leaver_about_actor": {"away"},
    "reality": set(), "belief_actor": set(),
}


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stories = generate(n_groups=120, seed=3)

    def test_labels_match_hand_table(self):
        for s, q in questions(self.stories):
            with self.subTest(q=q.qid):
                self.assertEqual(q.false_belief, s.variant in FB_TABLE[q.qtype])
                c1, c2 = s.containers
                if q.qtype == "reality":
                    self.assertEqual(q.answer, c1 if s.skeleton == "double_move" else c2)
                if q.false_belief:
                    # the stale belief is c1, except after the double move, where it is c2
                    self.assertEqual(q.answer, c2 if s.skeleton == "double_move" else c1)
                self.assertNotEqual(q.answer, q.foil)
                self.assertTrue(q.prompt.endswith("in the"))

    def test_variants_are_minimal_pairs(self):
        groups = {}
        for s in self.stories:
            groups.setdefault(s.group, []).append(s)
        for g, ss in groups.items():
            self.assertEqual(sorted(s.variant for s in ss), sorted(VARIANTS))
            base = ss[0]
            for other in ss[1:]:
                diff = [i for i, (a, b) in enumerate(zip(base.sentences, other.sentences)) if a != b]
                self.assertEqual(diff, [base.slot_index], g)
                self.assertEqual(len(base.sentences), len(other.sentences))

    def test_test_split_holds_out_combinations(self):
        key = lambda s: (s.names["leaver"], s.obj, frozenset(s.containers))
        train = {key(s) for s in self.stories if s.split == "train"}
        test = [s for s in self.stories if s.split == "test"]
        self.assertGreater(len(test), 0)
        self.assertTrue(all(key(s) not in train for s in test))
        words = lambda t: set(re.findall(r"[a-z']+", t.lower()))
        train_words = set().union(*(words(s.text) for s in self.stories if s.split == "train"))
        for s in test:                                   # new combinations, no new words
            self.assertLessEqual(words(s.text), train_words)

    def test_balance(self):
        c = Counter(s.skeleton for s in self.stories)
        self.assertLessEqual(max(c.values()) - min(c.values()), 3 * 2)
        fb = Counter(q.false_belief for _, q in questions(self.stories))
        self.assertGreater(fb[True], 0.25 * sum(fb.values()))

    def test_deterministic(self):
        a = generate(n_groups=10, seed=9)
        b = generate(n_groups=10, seed=9)
        self.assertEqual([s.text for s in a], [s.text for s in b])


if __name__ == "__main__":
    unittest.main()
