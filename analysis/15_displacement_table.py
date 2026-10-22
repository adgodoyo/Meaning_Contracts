"""
15_displacement_table.py — Refinement-minus-control displacement summary
(Table 14, tab:disp_summary) from the C1 cache of 02_refinement_distances.py.

For each model, readout and metric, the S0->S4 gap curve is
    diff(l) = gap_ref(l) - gap_com(l)
and the strict CI-separated fraction uses the conservative band
    lo(l) = (ref - ci_ref) - (com + ci_com)  >  0.

Input  : results/caches/refinement/{pooling}/{model}_{pooling}.npz   (C1 cache)
Output : results/tables/displacement_summary.csv
         columns: model, pooling, metric (euc|mah|cos), mean_gap, auc,
                  positive_frac, strict_positive_frac

Usage
-----
  python analysis/15_displacement_table.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import MODELS, TABLES_DIR, get_cache_dir

POOLINGS = ["mean", "last"]
METRIC_KEYS = ["euc", "mah", "cos"]      # cosine is the top-5-deflated version


def main() -> None:
    rows = []
    for model in MODELS:
        for pooling in POOLINGS:
            d = np.load(get_cache_dir("refinement", pooling) / f"{model}_{pooling}.npz", allow_pickle=True)
            for mk in METRIC_KEYS:
                ref, com = d[f"gap_ref_{mk}"][0, 4, :], d[f"gap_com_{mk}"][0, 4, :]
                ref_ci, com_ci = d[f"gap_ref_ci_{mk}"][0, 4, :], d[f"gap_com_ci_{mk}"][0, 4, :]
                diff = (ref - com).astype(np.float64)
                lo = (ref - ref_ci) - (com + com_ci)
                rows.append({
                    "model": model, "pooling": pooling, "metric": mk,
                    "mean_gap": f"{np.nanmean(diff):.6f}",
                    "auc": f"{np.nansum(diff):.6f}",
                    "positive_frac": f"{np.nanmean(diff > 0):.4f}",
                    "strict_positive_frac": f"{np.nanmean(lo > 0):.4f}",
                })

    out = TABLES_DIR / "displacement_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
