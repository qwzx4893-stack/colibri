"""High-Fidelity pipeline: find the cheapest quality sacrifice that reaches the daily-use speed, or prove none does.

Only used when Exact Mode is below the target (`agent.decide_exact_vs_high_fidelity`). Everything here is measured on the
real model through the real engine: no modelled speeds, no assumed quality.

  * `sensitivity()`  per-layer int4 sensitivity map (KL of the Exact distribution when only that layer's experts are int4)
  * `ladder()`       candidate profiles ordered by increasing expected quality loss: mixed int4 (least-sensitive layers
                     first), then dynamic expert sparsity, then their combination
  * `evaluate()`     steady-state speed + Quality Governor metrics (`quality.py`) for one profile
  * `pareto()`       non-dominated set over (speed, quality)
  * `select()`       Auto: the first profile in ladder order that reaches the target AND passes the quality budget
  * `search()`       the whole loop; stops at the first accepted profile (minimal quality loss to reach the target)
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import agent, quality


@dataclass
class Profile:
    name: str
    env: dict
    lossy: bool = True
    note: str = ""


def sensitivity(model_dir, workdir, qref, exact_dump, mask, n_layers, log=print, exe=None):
    """Per-layer int4 sensitivity: mean KL(Exact || layer-l-int4) over the answer positions. Returns {layer: metrics}."""
    out = {}
    for l in range(n_layers):
        path = Path(workdir) / f"sens_L{l}.qdump"
        quality.dump_logits(model_dir, {"LMA_GPU": "1", "LMA_HF_INT4": str(l)}, qref, path, exe)
        m = quality.compare_dumps(exact_dump, path, mask)
        path.unlink(missing_ok=True)
        out[l] = m
        log(f"  [sensitivity] layer {l:2d}: KL {m['kl_mean_nats']:.5f}  ppl x{m['ppl_ratio']:.4f}")
    return out


def ladder(sens=None, n_layers=16, topk=8):
    """Candidates from mildest to harshest. Each step adds either more int4 layers (in sensitivity order) or expert sparsity."""
    steps = [Profile("exact", {"LMA_GPU": "1"}, lossy=False)]
    if sens:
        order = sorted(sens, key=lambda l: sens[l]["kl_mean_nats"])
        for k in sorted({max(1, n_layers // 4), n_layers // 2, (3 * n_layers) // 4, n_layers}):
            steps.append(Profile(f"int4x{k}", {"LMA_GPU": "1", "LMA_HF_INT4": ",".join(str(l) for l in order[:k])},
                                 note=f"int4 group-32 experts on the {k} least sensitive layers"))
    for keep in range(topk - 1, max(1, topk // 2 - 1), -1):
        steps.append(Profile(f"keep{keep}", {"LMA_GPU": "1", "LMA_HF_KEEP": str(keep)}, note=f"top-{keep} of {topk} experts"))
    return steps


def evaluate(p, model_dir, workdir, ref_suite, exact_dump, qref, mask, log=print, exe=None):
    """Measure one profile: steady-state tok/s + task-suite comparison + logit divergence vs the Exact reference."""
    speed = agent.measure_steady_state_tok_s(model_dir, log=lambda *_: None, engine_env=p.env, reps=3)
    suite = quality.run_suite(model_dir, p.env)
    cmp_ = quality.compare_suites(ref_suite, suite)
    path = Path(workdir) / f"eval_{p.name}.qdump"
    quality.dump_logits(model_dir, p.env, qref, path, exe)
    logit = quality.compare_dumps(exact_dump, path, mask)
    path.unlink(missing_ok=True)
    v = quality.verdict(logit, cmp_)
    res = {"name": p.name, "env": p.env, "note": p.note, "lossy": p.lossy,
           "tok_s": speed["median_tok_s"] if speed else None, "tok_s_samples": speed["samples"] if speed else None,
           "logit": logit, "suite": cmp_["ALL"], "verdict": v}
    log(f"  [{p.name}] {res['tok_s']:.1f} tok/s  KL {logit['kl_mean_nats']:.4f}  top1 {logit['top1_agreement']:.3f}  "
        f"ppl x{logit['ppl_ratio']:.3f}  tasks {cmp_['ALL']['cand_ok']}/{cmp_['ALL']['ref_ok']}  -> "
        f"{'accepted' if v['accepted'] else 'rejected: ' + '; '.join(v['reasons'])}")
    return res


def pareto(evals):
    """Non-dominated points: faster AND closer to Exact (lower mean KL) than anything else that is not worse on both."""
    pts = [e for e in evals if e.get("tok_s") is not None]
    front = []
    for e in pts:
        dominated = any(o is not e and o["tok_s"] >= e["tok_s"] and o["logit"]["kl_mean_nats"] <= e["logit"]["kl_mean_nats"]
                        and (o["tok_s"] > e["tok_s"] or o["logit"]["kl_mean_nats"] < e["logit"]["kl_mean_nats"]) for o in pts)
        if not dominated:
            front.append(e)
    return sorted(front, key=lambda e: e["tok_s"])


def select(evals, target):
    """Auto: first profile in ladder order that reaches the target with an accepted quality verdict."""
    for e in evals:
        if e.get("tok_s") is not None and e["tok_s"] >= target and e["verdict"]["accepted"]:
            return e
    return None


def search(model_dir, workdir, target=None, budget=None, log=print, exe=None, n_layers=16, topk=8):
    """Run the full loop. Exact is measured first; nothing lossy is tried when it already reaches the target."""
    target = target or agent.DAILY_USE_TARGET_TOK_S
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    log(f"[hifi] measuring Exact against the {target:.0f} tok/s target")
    exact_speed = agent.measure_steady_state_tok_s(model_dir, log=lambda *_: None, engine_env={"LMA_GPU": "1"}, reps=3)
    if exact_speed and exact_speed["median_tok_s"] >= target:
        log(f"[hifi] Exact sustains {exact_speed['median_tok_s']:.1f} tok/s >= {target:.0f}: keeping Exact, no quality reduction")
        return {"mode": "exact", "target": target, "exact_tok_s": exact_speed["median_tok_s"], "selected": None, "pareto": [], "evaluated": []}
    log("[hifi] Exact is below target: building the Quality Governor reference (Exact answers + logits)")
    ref_suite = quality.run_suite(model_dir, {"LMA_GPU": "1"})
    qref = workdir / "qref.json"
    quality.build_qref(model_dir, ref_suite, qref, exe=exe)
    mask = str(qref) + ".mask.json"
    exact_dump = workdir / "exact.qdump"
    quality.dump_logits(model_dir, {"LMA_GPU": "1"}, qref, exact_dump, exe)
    log("[hifi] per-layer int4 sensitivity map")
    sens = sensitivity(model_dir, workdir, qref, exact_dump, mask, n_layers, log=log, exe=exe)
    evals, chosen = [], None
    for p in ladder(sens, n_layers, topk)[1:]:
        e = evaluate(p, model_dir, workdir, ref_suite, exact_dump, qref, mask, log=log, exe=exe)
        evals.append(e)
        if e["tok_s"] is not None and e["tok_s"] >= target and e["verdict"]["accepted"]:
            chosen = e; break                      # minimal-loss-first: stop at the first profile that gets there
    result = {"mode": "high_fidelity" if chosen else "exact_only", "target": target,
              "exact_tok_s": exact_speed["median_tok_s"] if exact_speed else None,
              "selected": chosen, "pareto": pareto(evals), "evaluated": evals,
              "sensitivity": {str(k): {"kl_mean_nats": v["kl_mean_nats"], "ppl_ratio": v["ppl_ratio"]} for k, v in sens.items()},
              "reason": ("selected " + chosen["name"]) if chosen else
                        "no profile reaches the target within the quality budget; Exact stays active (see pareto for the trade-offs)"}
    (workdir / "hifi_profiles.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"[hifi] {result['reason']}")
    return result
