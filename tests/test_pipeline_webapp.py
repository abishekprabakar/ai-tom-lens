"""End-to-end: analyse an (untrained) tiny model on a small corpus, then serve
the results through the Flask app and check every endpoint."""
import json
import math
import os
import tempfile
import unittest

import torch

from tomlens.generate import generate
from tomlens.tinylm import TinyLM, WordTokenizer, save_model
from tomlens.extract import TinyBackend
from tomlens.run import analyse, subsample


class TestPipelineAndWebapp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        stories = generate(n_groups=30, seed=5)
        tok = WordTokenizer.fit([q.prompt + " " + q.answer for s in stories for q in s.questions])
        torch.manual_seed(0)
        save_model(TinyLM("gpt", len(tok), d=32, layers=2, heads=4), tok, os.path.join(cls.tmp.name, "m.pt"))
        be = TinyBackend(os.path.join(cls.tmp.name, "m.pt"), name="toy")
        cls.results_dir = os.path.join(cls.tmp.name, "results")
        cls.res = analyse(be, stories, os.path.join(cls.results_dir, "toy"), log=lambda *a: None,
                          metric_steps=20, showcase_groups=3, topk=500)
        os.environ["TOMLENS_RESULTS"] = cls.results_dir
        import importlib
        import webapp.app as appmod
        cls.appmod = importlib.reload(appmod)
        cls.client = cls.appmod.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_results_shape(self):
        r = self.res
        self.assertEqual(r["n_layers"], 2)
        a = r["alignment"]
        self.assertEqual(len(a["raw"]["pooled"]), 3)
        self.assertEqual(len(a["learned"]), 3)
        for v in a["raw"]["pooled"] + a["learned"] + a["control"]:
            self.assertTrue(0 <= v <= 1)
        for lo, hi in a["raw_ci"]["pooled"]:
            self.assertLessEqual(lo, hi)
        self.assertIn("test|all", r["behaviour"])

    def test_api(self):
        models = self.client.get("/api/models").get_json()
        self.assertEqual([m["name"] for m in models], ["toy"])
        self.assertEqual(self.client.get("/").status_code, 200)
        mds = self.client.get("/api/mds/toy?layer=1").get_json()
        self.assertEqual(len(mds["cloud"]), len(mds["points"]))
        self.assertEqual(len(mds["centroids"]), 15)
        self.assertEqual(self.client.get("/api/mds/toy?layer=9").status_code, 400)
        self.assertEqual(self.client.get("/api/results/nope").status_code, 404)

    def test_token_endpoint_temperature(self):
        idx = self.client.get("/api/showcase/toy").get_json()
        qid = idx[0]["qid"]
        cold = self.client.get(f"/api/token/toy/{qid}?T=0.2&k=3").get_json()
        warm = self.client.get(f"/api/token/toy/{qid}?T=3&k=3").get_json()
        self.assertFalse(cold["renormalised_over_topk"])        # full vocab stored for the toy model
        for L in cold["layers"]:
            ps = [t["p"] for t in L["top"]]
            self.assertEqual(ps, sorted(ps, reverse=True))
        # lower temperature sharpens the distribution
        self.assertGreaterEqual(cold["layers"][-1]["top"][0]["p"], warm["layers"][-1]["top"][0]["p"])
        # answer probability at T=1 equals softmax of the stored logits over the full vocabulary
        one = self.client.get(f"/api/token/toy/{qid}?T=1").get_json()
        with open(os.path.join(self.results_dir, "toy", "viewer.json")) as f:
            sc = {s["qid"]: s for s in json.load(f)["showcase"]}[qid]
        L = sc["layers"][-1]
        z = sum(math.exp(l - max(L["logits"])) for l in L["logits"])
        p = math.exp(L["tracked"][0] - max(L["logits"])) / z
        self.assertAlmostEqual(one["layers"][-1]["answer_p"], p, places=4)

    def test_softmax_at(self):
        p, exact = self.appmod.softmax_at([2.0, 1.0, 0.0], 1.0, full_vocab=True)
        e = [math.exp(2), math.exp(1), 1]
        for a, b in zip(p, e):
            self.assertAlmostEqual(a, b / sum(e))
        p, exact = self.appmod.softmax_at([2.0, 1.0], 1.0, logsumexp=3.0)
        self.assertTrue(exact)
        self.assertAlmostEqual(p[0], math.exp(-1.0))


if __name__ == "__main__":
    unittest.main()
