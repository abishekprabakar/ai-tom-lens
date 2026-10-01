"""Triplet alignment, the triplet-loss gradient, metric learning and MDS,
each checked against something known independently of the implementation."""
import unittest

import numpy as np

from tomlens import alignment as al


class TestMDS(unittest.TestCase):
    def test_recovers_planar_configuration(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=(30, 2))
        D = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
        Z, stress = al.classical_mds(D)
        self.assertLess(stress, 1e-8)
        # Same configuration up to rotation/reflection/translation: orthogonal Procrustes.
        A, B = Y - Y.mean(0), Z - Z.mean(0)
        U, _, Vt = np.linalg.svd(B.T @ A)
        self.assertLess(np.abs(B @ U @ Vt - A).max(), 1e-8)

    def test_square_corners(self):
        # Unit square: classical MDS must return a square of side 1.
        D = np.array([[0, 1, np.sqrt(2), 1], [1, 0, 1, np.sqrt(2)],
                      [np.sqrt(2), 1, 0, 1], [1, np.sqrt(2), 1, 0]])
        Z, stress = al.classical_mds(D)
        Dz = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        np.testing.assert_allclose(Dz, D, atol=1e-10)

    def test_stress_positive_for_3d_data(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 3)) * [1, 1, 1]
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        _, stress = al.classical_mds(D)
        self.assertGreater(stress, 0.05)


class TestTripletLoss(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(2)
        W = rng.normal(size=(3, 5))
        A, P, N = (rng.normal(size=(20, 5)) for _ in range(3))
        _, g = al.triplet_loss_and_grad(W, A, P, N, margin=2.0)
        eps, num = 1e-6, np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Wp, Wm = W.copy(), W.copy()
                Wp[i, j] += eps; Wm[i, j] -= eps
                num[i, j] = (al.triplet_loss_and_grad(Wp, A, P, N, 2.0)[0]
                             - al.triplet_loss_and_grad(Wm, A, P, N, 2.0)[0]) / (2 * eps)
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-7)

    def test_zero_loss_when_satisfied(self):
        W = np.eye(2)
        A = np.zeros((1, 2)); P = np.zeros((1, 2)); N = np.array([[5.0, 0]])
        loss, g = al.triplet_loss_and_grad(W, A, P, N, margin=1.0)
        self.assertEqual(loss, 0.0)
        self.assertTrue(np.all(g == 0))


def toy_items(n_groups=80, seed=0):
    """Stories in 3-variant groups. Dim 0 is a big per-group 'surface' code
    shared by the twins; dim 1 is a small 'belief' signal; the rest is noise.
    Raw cosine is dominated by surface, so it should FAIL the triplets, and
    a learned metric should find dim 1 and pass them."""
    rng = np.random.default_rng(seed)
    items, X = [], []
    for g in range(n_groups):
        surf = rng.normal(0, 10, size=8)
        for v, fb in (("away", True), ("return", False), ("watch", False)):
            x = np.concatenate([surf, [1.0 if fb else -1.0], rng.normal(0, .3, 7)])
            items.append(dict(index=len(items), group=f"g{g}", qtype="belief_leaver",
                              variant=v, obj=f"o{g % 5}", false_belief=fb,
                              split="train" if g < n_groups * 0.7 else "test"))
            X.append(x)
    return items, np.array(X)


class TestTriplets(unittest.TestCase):
    def test_triplet_invariants(self):
        items, _ = toy_items()
        T = al.build_triplets(items)
        self.assertGreater(len(T), 0)
        for a, p, n in T:
            A, P, N = items[a], items[p], items[n]
            self.assertEqual(A["group"], N["group"])
            self.assertNotEqual(A["false_belief"], N["false_belief"])
            self.assertNotEqual(A["group"], P["group"])
            self.assertEqual(A["false_belief"], P["false_belief"])

    def test_binding_triplets(self):
        items = []
        for g in range(6):
            for v in ("away", "return", "watch"):
                for qt in ("belief_leaver", "belief_actor", "reality"):
                    items.append(dict(index=len(items), sid=f"g{g}-{v}", group=f"g{g}", variant=v, qtype=qt))
        T = al.build_binding_triplets(items)
        self.assertEqual(len(T), 6 * 2 * 4)
        for a, p, n in T:
            A, P, N = items[a], items[p], items[n]
            self.assertEqual(A["variant"], "away"); self.assertEqual(N["sid"], A["sid"])
            self.assertNotEqual(N["qtype"], A["qtype"]); self.assertEqual(P["qtype"], A["qtype"])
            self.assertNotEqual(P["group"], A["group"])

    def test_accuracy_hand_example(self):
        X = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [-1.0, 0]])
        T = np.array([[0, 1, 2], [0, 2, 1], [0, 1, 3]])
        self.assertAlmostEqual(al.triplet_accuracy(X, T), 2 / 3)

    def test_raw_fails_learned_passes(self):
        items, X = toy_items()
        tr = [i for i in items if i["split"] == "train"]
        te = [i for i in items if i["split"] == "test"]
        Ttr, Tte = al.build_triplets(tr, seed=1), al.build_triplets(te, seed=2)
        Xc = X - X.mean(0)
        self.assertLess(al.triplet_accuracy(Xc, Tte), 0.2)
        tr_idx = [i["index"] for i in tr]
        _, Z = al.standardize(X[tr_idx], X)
        W, losses = al.train_metric(Z, Ttr, k=4, steps=300, seed=0)
        self.assertLess(losses[-1], losses[0])
        self.assertGreater(al.triplet_accuracy(Z @ W.T, Tte, dist=al.euclid), 0.95)

    def test_bootstrap_ci_brackets_estimate(self):
        items, X = toy_items()
        T = al.build_triplets(items)
        groups = [i["group"] for i in items]
        acc = al.triplet_accuracy(X - X.mean(0), T)
        lo, hi = al.bootstrap_ci(X - X.mean(0), T, groups)
        self.assertLessEqual(lo, acc + 1e-12)
        self.assertGreaterEqual(hi, acc - 1e-12)

    def test_control_labels_are_a_function_of_surface(self):
        items, _ = toy_items()
        c = al.control_labels(items)
        table = {}
        for it, lab in zip(items, c):
            table.setdefault((it["obj"], it["variant"]), set()).add(lab)
        self.assertTrue(all(len(v) == 1 for v in table.values()))

    def test_standardize_pca_dims(self):
        rng = np.random.default_rng(0)
        Xtr, Xte = rng.normal(size=(300, 200)), rng.normal(size=(50, 200))
        Ztr, Zte = al.standardize(Xtr, Xte, max_dim=32)
        self.assertEqual(Ztr.shape, (300, 32))
        self.assertEqual(Zte.shape, (50, 32))
        np.testing.assert_allclose(Ztr.std(0), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
