"""
euclidean_deflation_sweep.py -- share of layers at which deflated Euclidean
distance (and, for comparison, cosine similarity) satisfies the orderings of
two_orderings.py, for deflation levels k = 5..10, both readouts (Table 13,
Appendix K).

Output
------
  results/robustness/euclidean_deflation_sweep.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, OUT_CSV_DIR, write_csv
from two_orderings import (  # noqa: E402
    INTERNAL, ordering_flags, per_pair, load_activations,
    load_multilingual_labels, build_multilingual_pair_indices, pairs_to_array,
)

KS = list(range(5, 11))
FLAGS = ("R", "Rm", "S", "S_plus")


def main() -> None:
    pl = build_multilingual_pair_indices(load_multilingual_labels())
    pairs = {c: pairs_to_array(pl[c]) for c in INTERNAL}

    counts: dict[tuple, dict[str, int]] = {}
    n_layers: dict[tuple, int] = {}
    for model in MODELS:
        for pooling in ("mean", "last"):
            acts = load_activations(model, "multilingual", pooling)
            N, L, D = acts.shape
            for layer in range(L):
                X = acts[:, layer, :].astype(np.float64)
                U, s, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
                for k in KS:
                    Xd = (U[:, k:] * s[k:]) @ Vt[k:]
                    for metric in ("eucl", "cos"):
                        vals = per_pair(Xd, pairs, metric)
                        kap = {c: float(v.mean()) for c, v in vals.items()}
                        f = ordering_flags(kap)
                        key = (model, pooling, metric, k)
                        c = counts.setdefault(key, {x: 0 for x in FLAGS})
                        for x in FLAGS:
                            c[x] += bool(f[x])
                        n_layers[key] = L
            print(f"  done {model} {pooling}", flush=True)

    rows = []
    for key, c in counts.items():
        model, pooling, metric, k = key
        n = n_layers[key]
        rows.append({"model": model, "pooling": pooling, "metric": metric, "k": k,
                     "n_layers": n, **{f"frac_{x}": round(c[x] / n, 4) for x in FLAGS}})
    write_csv(OUT_CSV_DIR / "euclidean_deflation_sweep.csv", rows)

    for metric in ("eucl", "cos"):
        print(f"\n% of layers satisfying S (S+ in brackets), {metric}")
        print(f"{'model':12s} {'pool':4s}" + "".join(f"{'k='+str(k):>11s}" for k in KS))
        for pooling in ("mean", "last"):
            for model in MODELS:
                cells = []
                for k in KS:
                    r = next(x for x in rows if (x["model"], x["pooling"], x["metric"], x["k"]) == (model, pooling, metric, k))
                    cells.append(f"{round(100*r['frac_S']):>4d} ({round(100*r['frac_S_plus']):>3d})")
                print(f"{model:12s} {pooling:4s}" + "".join(f"{c:>11s}" for c in cells))


if __name__ == "__main__":
    main()
