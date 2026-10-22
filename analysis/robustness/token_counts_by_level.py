"""
token_counts_by_level.py -- Table 17 (Appendix N): mean tokens per sentence at
each refinement level S0..S4, by arm and model, with each model's own tokenizer.

Input
-----
  results/caches/refinement/mean/{model}_n_tokens_constraints.npy
      per-sentence token counts written by analysis/02_refinement_distances.py
  data/constraints/constraints_sentences.jsonl   (arm, level)

Output
------
  results/robustness/token_counts_by_level.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FC_DATA, MODELS, OUT_CSV_DIR, RESULTS_CACHE, write_csv

ARMS = [("refinement", "Refinement"), ("commonality", "Control"),
        ("adv_near_synonym", "Near-synonym"), ("adv_scope_hedge", "Scope-hedge")]


def main() -> None:
    with open(FC_DATA / "constraints" / "constraints_sentences.jsonl") as f:
        recs = [json.loads(line) for line in f]
    arm = np.array([r["arm"] for r in recs])
    level = np.array([int(r["level"]) for r in recs])

    rows = []
    for model in MODELS:
        n_tok = np.load(RESULTS_CACHE / f"{model}_n_tokens_constraints.npy")
        for key, label in ARMS:
            means = [float(n_tok[(arm == key) & (level == lv)].mean()) for lv in range(5)]
            rows.append({"arm": label, "model": model,
                         **{f"S{lv}": round(m, 1) for lv, m in enumerate(means)},
                         "growth_S0_S4": round(means[4] - means[0], 1)})

    write_csv(OUT_CSV_DIR / "token_counts_by_level.csv", rows)
    print(f"\n{'arm':14s}{'model':14s}" + "".join(f"{'S'+str(i):>6s}" for i in range(5)) + f"{'S4-S0':>7s}")
    for r in rows:
        print(f"{r['arm']:14s}{r['model']:14s}" + "".join(f"{r['S'+str(i)]:6.1f}" for i in range(5))
              + f"{r['growth_S0_S4']:7.1f}")


if __name__ == "__main__":
    main()
