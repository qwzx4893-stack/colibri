"""`lmatomic auto`: Add Model -> Analyze -> Benchmark -> Optimize -> Ready, hiding nothing.

The result table always shows the Exact speed, the speed that will actually be used, and the measured quality retention
of the chosen profile against Exact (KL, top-1 agreement, perplexity ratio, task success on the Quality Governor suite).
"""
import json
import time
from pathlib import Path

from . import agent, hifi, profiles


def run(model_dir, target=None, log=print, rebuild=False, calibrate=True):
    model_dir = Path(model_dir)
    target = target or agent.DAILY_USE_TARGET_TOK_S
    t0 = time.time()
    log(f"[1/5] Add model      {model_dir}")
    graph = agent.inspect_model(model_dir)
    log(f"      family {graph['family']}, {graph['moe_layers']} MoE layers x {graph['experts_per_layer']} experts (top-{graph['topk']}), "
        f"{graph['total_bytes'] / 1e9:.1f} GB on disk")
    out = profiles.profile_dir(model_dir)
    prof = None if rebuild else profiles.load_profile(model_dir)
    if prof:
        log("[2/5] Analyze        reusing the stored profile (model and hardware unchanged)")
    else:
        log("[2/5] Analyze        hardware profile, routing calibration, preset tuning")
        prof = agent.build_agent(model_dir, out, calibrate=calibrate, log=lambda m: log("      " + str(m).strip()))
    pol = ((prof.get("strategy") or {}).get("performance_policy")) or {}
    exact = (pol.get("steady_state") or {}).get("median_tok_s")
    log(f"[3/5] Benchmark      Exact steady-state: {exact:.1f} tok/s (target {target:.0f})" if exact else
        "[3/5] Benchmark      Exact speed unavailable")
    selected = pol.get("selected_profile")
    if pol.get("mode") == "needs_high_fidelity" and not selected and exact is not None and exact < target:
        log("[4/5] Optimize       Exact is below target: High-Fidelity search under the quality budget")
        res = hifi.search(model_dir, out / "hifi", target=target, log=lambda m: log("      " + str(m).strip()))
        pol["mode"] = res["mode"] if res["mode"] != "exact_only" else "exact"
        pol["selected_profile"] = res["selected"]
        pol["pareto"] = [{k: e[k] for k in ("name", "tok_s", "env", "verdict")} | {"kl": e["logit"]["kl_mean_nats"]} for e in res["pareto"]]
        pol["hifi_reason"] = res["reason"]
        (out / "agent_profile.json").write_text(json.dumps(prof, indent=2), encoding="utf-8")
        selected = res["selected"]
    else:
        log("[4/5] Optimize       " + ("already optimised" if selected else "nothing to optimise: Exact meets the target"))
    log(f"[5/5] Ready          ({time.time() - t0:.0f} s)")
    rows = [("Exact speed", f"{exact:.1f} tok/s" if exact else "n/a")]
    if selected:
        lg, su = selected["logit"], selected["suite"]
        rows += [("Optimised speed", f"{selected['tok_s']:.1f} tok/s  ({selected['name']})"),
                 ("Quality retention", f"top-1 {lg['top1_agreement']:.3f}, KL {lg['kl_mean_nats']:.4f} nats, "
                                       f"ppl x{lg['ppl_ratio']:.3f}, tasks {su['cand_ok']}/{su['ref_ok']}")]
    else:
        rows += [("Optimised speed", "same as Exact (no lossy optimisation active)"), ("Quality retention", "100% (Exact, token-identical)")]
    for k, v in rows:
        log(f"      {k:<18} {v}")
    return prof
