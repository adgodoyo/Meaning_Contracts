"""
balanced_aggregation.py -- category means averaged over all pairs versus averaged
so that every factor, domain or language contributes equally (Section 3.2:
reweighting changes category means by less than 0.1%).

Output
------
  results/robustness/balanced_aggregation.csv
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FINAL_CODEBASE, MODELS, OUT_CSV_DIR, SEED,
    internal_to_paper, load_stable_band, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    PAIR_CATEGORIES, load_activations, load_multilingual_labels,
    build_multilingual_pair_indices, pairs_to_array,
)

CATS = list(PAIR_CATEGORIES.keys())
RANK = 128


def mahal_row_pairs(X: np.ndarray, pairs: np.ndarray, vecs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
    diff = (X[pairs[:, 0]].astype(np.float64) - X[pairs[:, 1]].astype(np.float64))
    proj = diff @ vecs * inv_sqrt
    return np.linalg.norm(proj, axis=1)


def group_keys_for_category(cat: str, pairs: np.ndarray, labels: dict) -> np.ndarray:
    """Return an array of group-key strings, one per pair, for balanced averaging."""
    fid, dom, lang = labels["factor_id"], labels["domain"], labels["language"]
    i, j = pairs[:, 0], pairs[:, 1]
    if cat in ("C1", "C2", "C3"):
        return fid[i]  # same factor on both sides -> group by factor
    if cat in ("C4", "C5"):
        return dom[i]  # same domain on both sides -> group by domain
    # C6, C7: different domains -> group by unordered domain pair
    di, dj = dom[i], dom[j]
    swap = di > dj
    lo = np.where(swap, dj, di)
    hi = np.where(swap, di, dj)
    return np.array([f"{a}|{b}" for a, b in zip(lo, hi)])


def balanced_mean(values: np.ndarray, keys: np.ndarray) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for v, k in zip(values, keys):
        groups[k].append(v)
    group_means = [float(np.mean(v)) for v in groups.values()]
    return float(np.mean(group_means))


def group_balance_diagnostic(keys: np.ndarray) -> float:
    """Coefficient of variation of pair-counts per group (0 = perfectly balanced sample)."""
    _, counts = np.unique(keys, return_counts=True)
    if len(counts) <= 1 or counts.mean() == 0:
        return 0.0
    return float(counts.std(ddof=0) / counts.mean())


def main() -> None:
    labels = load_multilingual_labels()
    pair_lists = build_multilingual_pair_indices(labels)
    pairs = {c: pairs_to_array(pair_lists[c]) for c in CATS}

    rows = []
    for model in MODELS:
        acts = load_activations(model, "multilingual", "mean")
        lo, hi = load_stable_band(model, "mean")
        target_layer = (lo + hi - 1) // 2  # stable-band midpoint
        X = acts[:, target_layer, :].astype(np.float64)
        Xc = X - X.mean(axis=0)
        N, D = Xc.shape
        k = min(RANK, N - 1, D - 1)
        np.random.seed(SEED)
        _, s, Vt = randomized_svd(Xc, n_components=k, random_state=SEED)
        eigvals = (s ** 2) / max(N - 1, 1)

        for cat in CATS:
            p = pairs[cat]
            if len(p) == 0:
                continue
            dists = mahal_row_pairs(Xc, p, Vt.T, eigvals)
            keys = group_keys_for_category(cat, p, labels)
            unbal = float(dists.mean())
            bal = balanced_mean(dists, keys)
            cv = group_balance_diagnostic(keys)
            paper_cat = internal_to_paper(cat)
            rows.append({
                "model": model, "layer": target_layer, "internal_cat": cat,
                "paper_cat": paper_cat, "n_pairs": len(p), "n_groups": len(np.unique(keys)),
                "group_count_cv": f"{cv:.4f}",
                "unbalanced_mean_dm": f"{unbal:.4f}",
                "balanced_mean_dm": f"{bal:.4f}",
                "abs_diff": f"{abs(bal - unbal):.4f}",
                "pct_diff": f"{100 * abs(bal - unbal) / max(abs(unbal), 1e-9):.2f}",
            })

    write_csv(OUT_CSV_DIR / "balanced_aggregation.csv", rows)

    max_pct = max(float(r["pct_diff"]) for r in rows)
    mean_pct = float(np.mean([float(r["pct_diff"]) for r in rows]))
    max_cv = max(float(r["group_count_cv"]) for r in rows)
    print(f"\nMax |balanced-unbalanced| relative difference: {max_pct:.2f}%")
    print(f"Mean relative difference across all model x category cells: {mean_pct:.2f}%")
    print(f"Max group-count coefficient of variation (sampling imbalance): {max_cv:.4f}")

    # Does the ordering (min/max category) change under balancing?
    order_changes = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        unbal_order = sorted(model_rows, key=lambda r: float(r["unbalanced_mean_dm"]))
        bal_order = sorted(model_rows, key=lambda r: float(r["balanced_mean_dm"]))
        unbal_seq = [r["paper_cat"] for r in unbal_order]
        bal_seq = [r["paper_cat"] for r in bal_order]
        order_changes.append((model, unbal_seq == bal_seq, unbal_seq, bal_seq))
        print(f"  {model}: ordering identical under balancing? {unbal_seq == bal_seq}")
        if unbal_seq != bal_seq:
            print(f"    unbalanced: {unbal_seq}")
            print(f"    balanced:   {bal_seq}")



if __name__ == "__main__":
    main()
