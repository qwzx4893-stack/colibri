import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lmatomic import hifi, profiles


def ev(name, tok_s, kl, accepted):
    return {"name": name, "tok_s": tok_s, "logit": {"kl_mean_nats": kl}, "verdict": {"accepted": accepted, "reasons": []}}


class Selection(unittest.TestCase):
    def test_select_takes_first_accepted_that_reaches_target(self):
        evals = [ev("a", 20, 0.001, True), ev("b", 35, 0.05, False), ev("c", 40, 0.01, True), ev("d", 90, 0.4, True)]
        self.assertEqual(hifi.select(evals, 30)["name"], "c")

    def test_select_none_when_nothing_qualifies(self):
        self.assertIsNone(hifi.select([ev("a", 20, 0.0, True), ev("b", 90, 0.5, False)], 30))

    def test_pareto_drops_dominated_points(self):
        front = hifi.pareto([ev("a", 20, 0.10, True), ev("b", 30, 0.05, True), ev("c", 40, 0.20, True), ev("d", 25, 0.08, True)])
        self.assertEqual([e["name"] for e in front], ["b", "c"])

    def test_ladder_is_mildest_first_and_starts_exact(self):
        sens = {l: {"kl_mean_nats": 0.01 * (16 - l)} for l in range(16)}   # layer 15 least sensitive
        steps = hifi.ladder(sens, 16, 8)
        self.assertEqual(steps[0].name, "exact")
        self.assertFalse(steps[0].lossy)
        first_int4 = next(p for p in steps if p.name.startswith("int4"))
        layers = [int(x) for x in first_int4.env["LMA_HF_INT4"].split(",")]
        self.assertEqual(layers, [15, 14, 13, 12])                        # the 4 least sensitive
        self.assertTrue(all(p.lossy for p in steps[1:]))
        self.assertTrue(any(p.name == "keep7" for p in steps))


class RuntimeEnv(unittest.TestCase):
    def test_runtime_env_from_profile(self):
        prof = {"strategy": {"performance_policy": {"cpu_baseline_tok_s": 3.9, "mode": "high_fidelity",
                                                    "selected_profile": {"env": {"LMA_HF_INT4": "1,2"}}}}}
        env = profiles.runtime_env(prof)
        self.assertEqual(env["LMA_CPU_BASELINE_TOK_S"], "3.9")
        self.assertEqual(env["LMA_HF_INT4"], "1,2")

    def test_exact_mode_never_injects_lossy_env(self):
        prof = {"strategy": {"performance_policy": {"mode": "exact", "selected_profile": {"env": {"LMA_HF_INT4": "all"}}}}}
        self.assertEqual(profiles.runtime_env(prof), {})
        self.assertEqual(profiles.runtime_env(None), {})

    def test_stale_or_missing_profile_is_not_used(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(profiles, "user_data_dir", return_value=Path(d)), \
                 mock.patch.object(profiles.agent, "model_hash", return_value="abc"):
                self.assertIsNone(profiles.load_profile("whatever"))
                pdir = profiles.profile_dir("whatever"); pdir.mkdir(parents=True)
                (pdir / "agent_profile.json").write_text(json.dumps({"schema": profiles.agent.SCHEMA, "model_hash": "OTHER"}))
                (pdir / "hardware_profile.json").write_text(json.dumps({"fingerprint_id": "x"}))
                self.assertIsNone(profiles.load_profile("whatever"))


if __name__ == "__main__":
    unittest.main()
