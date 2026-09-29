"""Digital twin and routing-fingerprint invariants (no engine, no GPU, no model).

These pin properties that must hold for any simulator result to be believable:
conservation of bytes, the NVMe roofline, monotonicity in cache size and locality, and
the ordering LRU <= Belady.
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lmatomic import routing, twin  # noqa: E402


def small_hw(vram_gb=0.0, ram_gb=0.0, nvme=3.0):
    return twin.Hardware(
        nvme_grid={1: [(1, nvme * 0.5), (64, nvme * 0.9)], 8: [(1, nvme * 0.8), (64, nvme)]},
        h2d_gbs=12.0, ram_gbs=40.0, cpu_gbs=30.0, vram_gbs=250.0, gpu_expert_ms=0.05,
        gpu_expert_bytes=6.3e6, vram_cache_bytes=vram_gb * 1e9, ram_cache_bytes=ram_gb * 1e9)


def spec(layers=8, experts=64, topk=4, expert_mb=8):
    return twin.ModelSpec("t", layers, experts, topk, expert_mb * 1e6, 0.0)


class Routing(unittest.TestCase):
    def test_lru_never_beats_belady(self):
        rng = random.Random(3)
        seq = [rng.sample(range(32), 4) for _ in range(400)]
        for cap in (2, 4, 8, 16):
            self.assertLessEqual(routing.lru_hit(seq, cap), routing.belady_hit(seq, cap) + 1e-9)

    def test_full_cache_hits_everything_after_warmup(self):
        seq = [[1, 2, 3, 4]] * 50
        self.assertGreater(routing.lru_hit(seq, 4), 0.95)

    def test_trace_roundtrip_and_fingerprint(self):
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            call = 0
            with open(p, "w") as f:
                prev = [rng.sample(range(16), 3) for _ in range(4)]
                for tok in range(60):
                    for layer in range(4):
                        ids = [e for e in prev[layer] if rng.random() < 0.6]
                        while len(ids) < 3:
                            e = rng.randrange(16)
                            if e not in ids:
                                ids.append(e)
                        prev[layer] = ids
                        f.write(f"{call} 0 {layer} " + " ".join(f"{e}:0.3333" for e in ids) + "\n")
                        call += 1
            tr = routing.parse(p, 16)
            self.assertEqual(tr.n_layers, 4)
            self.assertEqual(len(list(tr.decode_tokens())), 60)
            fp = routing.fingerprint(tr)
            self.assertGreater(fp["reuse_next_token"], fp["chance_next_token"])
            for c in fp["cache_curves"].values():
                self.assertLessEqual(c["lru_hit"], c["belady_hit"] + 1e-9)

    def test_prefill_forwards_are_not_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            p.write_text("0 0 0 1:0.5 2:0.5\n0 1 0 3:0.5 4:0.5\n1 0 1 1:0.5 2:0.5\n1 1 1 3:0.5 4:0.5\n")
            tr = routing.parse(p, 8)
            self.assertEqual(len(list(tr.decode_tokens())), 0)
            self.assertEqual(len(list(tr.prefill_forwards())), 1)


class Twin(unittest.TestCase):
    def run_sim(self, sp, hw, preset, policy, tokens=200, seed=1):
        return twin.simulate_decode(sp, hw, iter(lambda g=twin.SyntheticRouting(sp, preset, seed=seed): g.token(), None),
                                    policy, tokens)

    def test_everything_resident_means_no_nvme(self):
        sp = spec(layers=4, experts=16, topk=4, expert_mb=4)
        hw = small_hw(ram_gb=sp.routed_bytes / 1e9 * 1.5)
        r = self.run_sim(sp, hw, "typical", twin.Policy(warm_tokens=100), tokens=300)
        self.assertEqual(r["nvme_gb_per_token"], 0.0)
        self.assertAlmostEqual(r["miss"], 0.0)

    def test_nvme_roofline(self):
        """tokens/s can never exceed NVMe bandwidth / NVMe bytes per token."""
        sp = spec(layers=8, experts=64, topk=4, expert_mb=8)
        hw = small_hw(ram_gb=0.2)
        r = self.run_sim(sp, hw, "hostile", twin.Policy(warm_tokens=20), tokens=120)
        peak = max(g for pts in hw.nvme_grid.values() for _, g in pts)
        self.assertLessEqual(r["tok_s"], peak / r["nvme_gb_per_token"] * 1.001)

    def test_bytes_per_token_bounded_by_uncached(self):
        sp = spec()
        hw = small_hw(ram_gb=0.3)
        r = self.run_sim(sp, hw, "hostile", twin.Policy(warm_tokens=20), tokens=100)
        self.assertLessEqual(r["nvme_gb_per_token"], sp.bytes_per_token_uncached / 1e9 + 1e-9)

    def test_locality_helps_and_cache_helps(self):
        sp = spec()
        hits = {}
        for preset in ("hostile", "typical", "favorable"):
            hits[preset] = 1 - self.run_sim(sp, small_hw(ram_gb=0.6), preset, twin.Policy(warm_tokens=60), tokens=300)["miss"]
        self.assertLess(hits["hostile"], hits["typical"])
        self.assertLess(hits["typical"], hits["favorable"])
        small = self.run_sim(sp, small_hw(ram_gb=0.3), "typical", twin.Policy(warm_tokens=60), tokens=300)["miss"]
        big = self.run_sim(sp, small_hw(ram_gb=1.2), "typical", twin.Policy(warm_tokens=60), tokens=300)["miss"]
        self.assertLess(big, small)

    def test_prefetch_metrics_are_probabilities(self):
        # Small experts, fast NVMe, deep lookahead: the read fits inside the prefetch
        # window. (A slow NVMe / huge-expert / depth=1 combination can make prefetch
        # physically infeasible -- see test_prefetch_infeasible_issues_nothing -- that
        # is a correct twin result, not a bug, so this test picks a feasible regime.)
        sp = spec(layers=8, experts=64, topk=4, expert_mb=1)
        hw = small_hw(ram_gb=0.6, nvme=40.0)
        pol = twin.Policy(name="pf", prefetch=True, depth=3, delta=2, byte_budget_frac=0.9, warm_tokens=60)
        r = self.run_sim(sp, hw, "favorable", pol, tokens=300)
        self.assertGreater(r["prefetch_issued"], 0)
        self.assertGreaterEqual(r["prefetch_useful_frac"], 0.0)
        self.assertLessEqual(r["prefetch_useful_frac"], 1.0)

    def test_prefetch_infeasible_issues_nothing(self):
        # Expert reads far slower than one layer's compute (true at real scale too: a
        # 13-25 MB expert read costs 3-6 ms, a decode layer computes in ~0.1-1 ms) mean
        # a depth-1 prefetch cannot land in time. The scheduler must not pretend
        # otherwise by issuing reads it cannot finish.
        sp = spec(layers=8, experts=64, topk=4, expert_mb=8)
        hw = small_hw(ram_gb=0.6, nvme=3.0)
        pol = twin.Policy(name="pf", prefetch=True, depth=1, delta=2, warm_tokens=60)
        r = self.run_sim(sp, hw, "favorable", pol, tokens=300)
        self.assertEqual(r["prefetch_issued"], 0)

    def test_giant_specs_add_up(self):
        for key in twin.GIANTS:
            g = twin.make_giant(key)
            self.assertAlmostEqual(g.layers * g.experts * g.expert_bytes / (4.5 / 8) + g.dense_bytes / (4.5 / 8), g.params_total, delta=g.params_total * 1e-6)
            self.assertLess(g.params_active, g.params_total)


if __name__ == "__main__":
    unittest.main()
