import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from lmatomic import quality as Q


def _write_dump(path, targets, logprobs):
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", logprobs.shape[1], len(targets)))
        for t, row in zip(targets, logprobs):
            f.write(struct.pack("<i", int(t)))
            f.write(row.astype(np.float32).tobytes())


def _logsoftmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


class QualityMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        self.logits = rng.normal(size=(40, 50)) * 3
        self.tgt = self.logits.argmax(1)
        self.ref = Path(self.tmp.name) / "ref.bin"
        _write_dump(self.ref, self.tgt, _logsoftmax(self.logits))

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_dump_has_zero_divergence(self):
        m = Q.compare_dumps(self.ref, self.ref)
        self.assertEqual(m["positions"], 40)
        self.assertEqual(m["top1_agreement"], 1.0)
        self.assertAlmostEqual(m["kl_mean_nats"], 0.0, places=6)
        self.assertAlmostEqual(m["ppl_ratio"], 1.0, places=6)
        self.assertTrue(Q.verdict(m, None)["accepted"])

    def test_perturbed_dump_is_measured_and_rejected(self):
        cand = Path(self.tmp.name) / "cand.bin"
        noisy = self.logits + np.random.default_rng(9).normal(size=self.logits.shape) * 2.5
        _write_dump(cand, self.tgt, _logsoftmax(noisy))
        m = Q.compare_dumps(self.ref, cand)
        self.assertLess(m["top1_agreement"], 0.97)
        self.assertGreater(m["kl_mean_nats"], 0.02)
        v = Q.verdict(m, None)
        self.assertFalse(v["accepted"])
        self.assertTrue(v["reasons"])

    def test_mask_restricts_positions(self):
        mask = Path(self.tmp.name) / "m.json"
        mask.write_text(json.dumps([1] * 10 + [0] * 30))
        m = Q.compare_dumps(self.ref, self.ref, mask)
        self.assertEqual(m["positions"], 10)


class SuiteComparison(unittest.TestCase):
    def test_success_retention_and_agreement(self):
        ref = [{"domain": "math", "text": "the answer is 41", "ok": True},
               {"domain": "math", "text": "56", "ok": True},
               {"domain": "factual", "text": "Paris", "ok": True}]
        cand = [{"domain": "math", "text": "the answer is 42", "ok": False},
                {"domain": "math", "text": "56", "ok": True},
                {"domain": "factual", "text": "Paris", "ok": True}]
        c = Q.compare_suites(ref, cand)
        self.assertEqual(c["ALL"]["identical"], 2)
        self.assertAlmostEqual(c["ALL"]["success_retention"], 2 / 3, places=3)
        self.assertAlmostEqual(c["math"]["prefix_agree"], (0.75 + 1.0) / 2, places=3)
        self.assertFalse(Q.verdict(None, c)["accepted"])

    def test_checkers_accept_and_reject(self):
        by = {p[:30]: chk for _, p, chk in Q.SUITE}
        chk = [c for _, p, c in Q.SUITE if p.startswith("What is 12 + 29")][0]
        self.assertTrue(chk("It is 41."))
        self.assertFalse(chk("It is 42."))
        js = [c for _, p, c in Q.SUITE if "Ana" in p][0]
        self.assertTrue(js('Sure: {"name": "Ana", "age": 30}'))
        self.assertFalse(js("no json here"))
        code = [c for _, p, c in Q.SUITE if "square(x)" in p][0]
        self.assertTrue(code("```python\ndef square(x):\n    return x * x\n```"))
        self.assertFalse(code("```python\ndef square(x):\n    return x + x\n```"))


if __name__ == "__main__":
    unittest.main()
