import json, statistics, sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from ab_bench import run_one, CONFIGS

NEW = r"C:\Users\STS\projects\colibri\c\olmoe.exe"

if __name__ == "__main__":
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    prompt_key = sys.argv[2] if len(sys.argv) > 2 else "short"
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    all_runs = {}
    texts = {}
    cfg_names = sys.argv[5].split(",") if len(sys.argv) > 5 else ["pilot_old_equiv", "pilot"]
    for cfg_name in cfg_names:
        runs = []
        for i in range(reps):
            r = run_one(NEW, cfg_name, prompt_key, max_tokens, extra_env=CONFIGS[cfg_name], cap=cap)
            runs.append(r)
            print(f"[{cfg_name} rep {i+1}/{reps}] wall_s={r['wall_s']:.3f} stat={r['stat']}")
        all_runs[cfg_name] = runs
        texts[cfg_name] = runs[0]["text"]

    print()
    summary = {}
    for cfg_name, runs in all_runs.items():
        tok_s = [r["stat"]["engine_tok_s"] for r in runs if r["stat"]]
        gbtok = [r["stat"]["gb_per_token"] for r in runs if r["stat"]]
        prec = [r["stat"]["pf_precision"] for r in runs if r["stat"]]
        wall = [r["wall_s"] for r in runs]
        summary[cfg_name] = {
            "median_tok_s": statistics.median(tok_s) if tok_s else None,
            "median_gb_per_token": statistics.median(gbtok) if gbtok else None,
            "median_pf_precision": statistics.median(prec) if prec else None,
            "median_wall_s": statistics.median(wall),
            "all_tok_s": tok_s, "all_gb_per_token": gbtok, "all_wall_s": wall,
        }
        print(f"=== {cfg_name} (n={reps}) === median tok/s={summary[cfg_name]['median_tok_s']} "
              f"median GB/token={summary[cfg_name]['median_gb_per_token']} "
              f"median pf_precision={summary[cfg_name]['median_pf_precision']} "
              f"median wall_s={summary[cfg_name]['median_wall_s']:.2f}")

    all_texts = set(texts.values())
    print(f"\nEXACT MATCH across all configs: {len(all_texts) == 1}")
    json.dump({"summary": summary, "texts": texts}, open(r"C:\Users\STS\lma_ab\ablation_result.json", "w"), indent=2)
