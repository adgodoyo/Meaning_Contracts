"""
two_orderings.py -- the relatedness ordering R and the core-and-spread ordering
B (called S in the output columns), tested on every metric and readout
(Section 3.2, Table 7; the cosine counts reproduce Table 8).

Closeness kappa is cosine similarity for cosine and -distance for Euclidean and
Mahalanobis distance, so larger always means closer. Paper category labels.

  R   min G1 > max G2 and min G2 > max G3,
      G1={C1,C2,C3}, G2={C4,C5}, G3={C6,C7}
  Rm  group-mean version of R
  S   kappa_C1 > max M and min M > max(kappa_C2, kappa_C3), M={C4..C7}
  S+  S and min(C4,C5) > max(C6,C7)

Every (model, readout, metric, condition) is evaluated at every layer, raw and
at the deflation level used in the figures. Deflation and the Mahalanobis
eigenspace come from one exact SVD of the centred N x D matrix per layer.
Mahalanobis retains r=128 (mean) and r=256 (last token), plus r=512 for the
last token.

Outputs
-------
  results/robustness/two_orderings_layers.csv
  results/robustness/two_orderings_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FINAL_CODEBASE, MODELS, OUT_CSV_DIR, internal_to_paper, write_csv

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    PAIR_CATEGORIES, load_activations, load_multilingual_labels,
    build_multilingual_pair_indices, pairs_to_array,
)

INTERNAL = list(PAIR_CATEGORIES.keys())
CATS = [f"C{i}" for i in range(1, 8)]

# Deflation level used in the paper's figures (Section 4.1).
FIG_K = {
    ("pythia-2.8b", "mean"): 8, ("llama3.2-3b", "mean"): 5,
    ("qwen2.5-3b", "mean"): 5,  ("gemma4-e4b", "mean"): 5,
    ("pythia-2.8b", "last"): 10, ("llama3.2-3b", "last"): 4,
    ("qwen2.5-3b", "last"): 4,  ("gemma4-e4b", "last"): 4,
}
MAHAL_R = {"mean": [128], "last": [256, 512]}

# Table 8 of the paper: cosine, mean pooling, layers satisfying strict R, by k.
TABLE8 = {
    "pythia-2.8b": {0: 0, 4: 2, 5: 0, 6: 0, 7: 4, 8: 7, 9: 8, 10: 15},
    "llama3.2-3b": {0: 0, 4: 6, 5: 17, 6: 19, 7: 21, 8: 22, 9: 22, 10: 24},
    "qwen2.5-3b":  {0: 0, 4: 17, 5: 28, 6: 28, 7: 28, 8: 27, 9: 27, 10: 28},
    "gemma4-e4b":  {0: 2, 4: 12, 5: 8, 6: 6, 7: 9, 8: 8, 9: 15, 10: 17},
}


# ----------------------------------------------------------------- orderings
def ordering_flags(kap: dict[str, float]) -> dict[str, bool]:
    g1 = [kap[c] for c in ("C1", "C2", "C3")]
    g2 = [kap[c] for c in ("C4", "C5")]
    g3 = [kap[c] for c in ("C6", "C7")]
    mid = [kap[c] for c in ("C4", "C5", "C6", "C7")]
    S = kap["C1"] > max(mid) and min(mid) > max(kap["C2"], kap["C3"])
    return {
        "R": min(g1) > max(g2) and min(g2) > max(g3),
        "Rm": np.mean(g1) > np.mean(g2) > np.mean(g3),
        "S": S,
        "S_plus": S and min(g2) > max(g3),
    }


def category_means(values: np.ndarray, pairs: dict) -> dict[str, float]:
    """values: per-pair closeness for the concatenated pair list."""
    return {c: float(values[c].mean()) for c in values}


# ------------------------------------------------------------------- metrics
def per_pair(X: np.ndarray, pairs: dict, metric: str) -> dict[str, np.ndarray]:
    out = {}
    for ic in INTERNAL:
        p = pairs[ic]
        a, b = X[p[:, 0]], X[p[:, 1]]
        if metric == "cos":
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            v = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
        else:  # eucl / mahal: closeness = -distance
            v = -np.linalg.norm(a - b, axis=1)
        out[internal_to_paper(ic)] = v
    return out


def main(models: list[str], tag: str = "") -> None:
    labels = load_multilingual_labels()
    pl = build_multilingual_pair_indices(labels)
    pairs = {c: pairs_to_array(pl[c]) for c in INTERNAL}

    layer_rows = []
    for model in models:
        for pooling in ("mean", "last"):
            acts = load_activations(model, "multilingual", pooling)
            N, L, D = acts.shape
            kfig = FIG_K[(model, pooling)]
            ks = sorted({0, kfig} | (set(range(4, 11)) if pooling == "mean" else set()))
            for layer in range(L):
                X = acts[:, layer, :].astype(np.float64)
                Xc = X - X.mean(axis=0)
                U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
                for k in ks:
                    # Deflated, centred data = SVD components k..; raw cosine and
                    # Euclidean use the uncentred states, as in the paper's raw branch.
                    Xd = X if k == 0 else (U[:, k:] * s[k:]) @ Vt[k:]
                    metrics = [("cos", Xd), ("eucl", Xd)]
                    for r in MAHAL_R[pooling]:
                        comp = slice(k, k + r)
                        lam = s[comp] ** 2 / (N - 1)
                        Z = (U[:, comp] * s[comp]) / np.sqrt(np.maximum(lam, 1e-10))
                        metrics.append((f"mahal_r{r}", Z))
                    for mname, M in metrics:
                        if k not in (0, kfig) and not (mname == "cos" and pooling == "mean"):
                            continue
                        kap = category_means(per_pair(M, pairs, mname.split("_")[0]), pairs)
                        flags = ordering_flags(kap)
                        layer_rows.append({
                            "model": model, "pooling": pooling, "metric": mname,
                            "k": k, "is_fig_k": k == kfig, "layer": layer,
                            **{f"kappa_{c}": round(kap[c], 6) for c in CATS},
                            **flags,
                        })
            print(f"  done {model} {pooling} (L={L})", flush=True)

    write_csv(OUT_CSV_DIR / f"two_orderings_layers{tag}.csv", layer_rows)

    # ---- check against Table 8 of the paper ----
    print("\nCheck: cosine / mean / strict R qualifying layers vs Table 8")
    ok = True
    for model in models:
        for k, expected in TABLE8[model].items():
            n = sum(r["R"] for r in layer_rows if r["model"] == model and r["pooling"] == "mean"
                    and r["metric"] == "cos" and r["k"] == k)
            flag = "" if n == expected else "   <-- differs"
            ok &= n == expected
            print(f"  {model:12s} k={k:2d}  table8={expected:2d}  recomputed={n:2d}{flag}")
    print("  ALL MATCH" if ok else "  MISMATCHES PRESENT")

    # ---- summary: fraction of layers per ordering ----
    summary = []
    keys = sorted({(r["model"], r["pooling"], r["metric"], r["k"]) for r in layer_rows})
    for model, pooling, metric, k in keys:
        if k not in (0, FIG_K[(model, pooling)]):
            continue
        rows = [r for r in layer_rows if (r["model"], r["pooling"], r["metric"], r["k"]) == (model, pooling, metric, k)]
        n = len(rows)
        summary.append({
            "model": model, "pooling": pooling, "metric": metric,
            "condition": "raw" if k == 0 else f"top{k}", "n_layers": n,
            **{f"frac_{f}": round(sum(r[f] for r in rows) / n, 4) for f in ("R", "Rm", "S", "S_plus")},
        })
    write_csv(OUT_CSV_DIR / f"two_orderings_summary{tag}.csv", summary)

    print(f"\n{'model':12s} {'pool':4s} {'metric':11s} {'cond':6s} {'R':>5} {'Rm':>5} {'S':>5} {'S+':>5}")
    for s_ in summary:
        print(f"{s_['model']:12s} {s_['pooling']:4s} {s_['metric']:11s} {s_['condition']:6s} "
              f"{s_['frac_R']:5.2f} {s_['frac_Rm']:5.2f} {s_['frac_S']:5.2f} {s_['frac_S_plus']:5.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = ap.parse_args()
    # A subset run writes to suffixed files so it never overwrites the full run.
    main(args.models, "" if args.models == MODELS else "_" + "_".join(args.models))
