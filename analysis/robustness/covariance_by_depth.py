"""
covariance_by_depth.py -- the Mahalanobis ordering with the covariance
estimated on sentence populations other than the ones measured (Appendix F).

Estimation sets: the full probe (reference); disjoint halves of the 40 factors;
five of the ten domains; the European languages; random draws of 300 and 360
sentences; and the refinement probe (external corpus, with the probe's own
top-k deflation basis retained). Pairs are always measured on sentences not used
for estimation. Evaluated at normalised depths 0.4, 0.5, 0.6, 0.7 for both
readouts. Per cell: C1 nearest (B1), middle band below C2/C3 (B2), band means
ordered (B3), B1 and B2 (the core-and-spread ordering B), and the normalised
margin. The paper reports normalised depth 0.5.

Outputs
-------
  results/robustness/covariance_by_depth_cells.csv
  results/robustness/covariance_by_depth_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FINAL_CODEBASE, MODELS, OUT_CSV_DIR, SEED,
    build_constraints_activation_index, internal_to_paper,
    load_kmin, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    PAIR_CATEGORIES, load_activations, load_multilingual_labels,
    build_multilingual_pair_indices, pairs_to_array, normalised_depth_array,
)

CATS = list(PAIR_CATEGORIES.keys())
PAPER_CATS = [f"C{i}" for i in range(1, 8)]
MID = ("C4", "C5", "C6", "C7")
FAR = ("C2", "C3")
PAPER_RANK = {"mean": 128, "last": 256}
DEPTHS = [0.4, 0.5, 0.6, 0.7]
N_DRAWS = 10
EURO = ("en", "de", "es")


def top_k_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of the centred data, shape (D, k). Exact."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T


def deflate(X: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return Xc - (Xc @ Vk) @ Vk.T


def eig_top(X: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-`rank` eigenpairs of cov(X), via thin SVD of the centred data.

    Mathematically identical to eigh on the D x D covariance for the leading
    min(n-1, D) directions, but O(n^2 D) instead of O(D^3). Here n is 300-3,840
    and D is 2,048-3,072, and rank (128 or 256) is always below n-1, so the thin
    SVD returns exactly the directions needed (LAPACK's exact SVD, not a
    randomized approximation).
    """
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    k = min(rank, n - 1, Xc.shape[1])
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    vals = (s[:k] ** 2) / max(n - 1, 1)
    return Vt[:k].T, vals


def cat_means(X: np.ndarray, keep: dict, vecs: np.ndarray,
              vals: np.ndarray) -> dict[str, float]:
    inv = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
    out = {}
    for c in CATS:
        p = keep[c]
        pc = internal_to_paper(c)
        out[pc] = (float(np.linalg.norm((X[p[:, 0]] - X[p[:, 1]]) @ vecs * inv, axis=1).mean())
                   if len(p) else np.nan)
    return out


def tiers(d: dict[str, float]) -> dict:
    core = d["C1"]
    mid = [d[c] for c in MID]
    far = [d[c] for c in FAR]
    span = min(far) - core
    b1, b2 = core < min(mid), max(mid) < min(far)
    return {
        "B1_core_below_mid": b1,
        "B2_mid_below_far": b2,
        "B3_tier_means_ordered": core < float(np.mean(mid)) < float(np.mean(far)),
        "B1_and_B2": b1 and b2,
        "margin_B2_norm": f"{(min(far) - max(mid)) / span:.4f}" if span else "",
        **{f"dM_{c}": f"{d[c]:.4f}" for c in PAPER_CATS},
    }


def main() -> None:
    labels = load_multilingual_labels()
    factor = np.asarray(labels["factor_id"], dtype=object)
    dom = np.asarray(labels["domain"], dtype=object)
    lang = np.asarray(labels["language"], dtype=object)

    in_A = np.array([f[-1] in ("1", "2") for f in factor])
    in_B = ~in_A
    assert in_A.sum() == in_B.sum() == 600

    pair_lists = build_multilingual_pair_indices(labels)
    pairs_all = {c: pairs_to_array(pair_lists[c]) for c in CATS}
    keep_B = {c: pairs_all[c][in_B[pairs_all[c][:, 0]] & in_B[pairs_all[c][:, 1]]]
              for c in CATS}
    dom5 = sorted(set(dom))[:5]
    idx_A = np.flatnonzero(in_A)

    rows: list[dict] = []
    for model in MODELS:
        idx_cn = build_constraints_activation_index(model)
        for pooling in ("mean", "last"):
            rank = PAPER_RANK[pooling]
            k = load_kmin(model, pooling)
            acts_ml = load_activations(model, "multilingual", pooling)
            acts_cn = load_activations(model, "constraints", pooling)
            L = acts_ml.shape[1]
            nd = normalised_depth_array(L)

            for target in DEPTHS:
                layer = int(np.argmin(np.abs(nd - target)))
                X = acts_ml[:, layer, :].astype(np.float64)
                Vk = top_k_basis(X, k)
                Xd = deflate(X, Vk)

                def emit(name: str, sel: np.ndarray, keep: dict, cov: str,
                         ref: list[float] | None, trial, n_cov: int) -> list[float]:
                    v, l = eig_top(Xd[sel] if cov == "probe" else sel_ext, rank)
                    d = cat_means(Xd, keep, v, l)
                    obs = [d[c] for c in PAPER_CATS]
                    rho = "" if ref is None else f"{float(spearmanr(obs, ref).statistic):.4f}"
                    rows.append({
                        "model": model, "pooling": pooling,
                        "target_nd": target, "layer": layer,
                        "actual_nd": round(float(nd[layer]), 4),
                        "deflation_k": k, "rank": rank,
                        "estimation_set": name, "trial": trial,
                        "n_cov_samples": n_cov,
                        "eval_set": "factor_half_B" if keep is keep_B else "all_pairs",
                        "rho_vs_ref": rho, **tiers(d),
                    })
                    return obs

                # ---- evaluated on ALL pairs ----
                ref_all = emit("own_full", np.arange(X.shape[0]), pairs_all,
                               "probe", None, "fixed", X.shape[0])
                rows[-1]["rho_vs_ref"] = "1.0000"
                Xcn = deflate(acts_cn[idx_cn][:, layer, :].astype(np.float64), Vk)
                sel_ext = Xcn
                emit("external_shared_deflation", None, pairs_all, "ext",
                     ref_all, "fixed", Xcn.shape[0])

                # ---- evaluated on factor-half B pairs ----
                ref_B = emit("self_B", np.flatnonzero(in_B), keep_B,
                             "probe", None, "fixed", 600)
                rows[-1]["rho_vs_ref"] = "1.0000"
                emit("A_full", idx_A, keep_B, "probe", ref_B, "fixed", len(idx_A))
                s5 = idx_A[np.isin(dom[idx_A], dom5)]
                se = idx_A[np.isin(lang[idx_A], EURO)]
                emit("A_5domains", s5, keep_B, "probe", ref_B, "fixed", len(s5))
                emit("A_eurolang", se, keep_B, "probe", ref_B, "fixed", len(se))
                rng = np.random.default_rng(SEED)
                for t in range(N_DRAWS):
                    emit(f"A_rand{len(s5)}", rng.choice(idx_A, len(s5), replace=False),
                         keep_B, "probe", ref_B, t, len(s5))
                for t in range(N_DRAWS):
                    emit(f"A_rand{len(se)}", rng.choice(idx_A, len(se), replace=False),
                         keep_B, "probe", ref_B, t, len(se))
                print(f"  {model:12s} {pooling:4s} ND={target} (L{layer:2d}) done", flush=True)
            del acts_ml, acts_cn

    write_csv(OUT_CSV_DIR / "covariance_by_depth_cells.csv", rows)

    # ---- summary ----
    names = ["own_full", "external_shared_deflation", "self_B", "A_full",
             "A_5domains", "A_eurolang"]
    names += sorted({r["estimation_set"] for r in rows
                     if r["estimation_set"].startswith("A_rand")})
    summary: list[dict] = []
    for pooling in ("mean", "last"):
        for target in DEPTHS:
            for name in names:
                sub = [r for r in rows if r["pooling"] == pooling
                       and r["target_nd"] == target and r["estimation_set"] == name]
                if not sub:
                    continue
                m2 = [float(r["margin_B2_norm"]) for r in sub if r["margin_B2_norm"]]
                rh = [float(r["rho_vs_ref"]) for r in sub if r["rho_vs_ref"]]
                summary.append({
                    "pooling": pooling, "target_nd": target, "estimation_set": name,
                    "n_cov_samples": sub[0]["n_cov_samples"], "n_cells": len(sub),
                    "B1": f"{sum(r['B1_core_below_mid'] for r in sub)}/{len(sub)}",
                    "B2": f"{sum(r['B2_mid_below_far'] for r in sub)}/{len(sub)}",
                    "B3": f"{sum(r['B3_tier_means_ordered'] for r in sub)}/{len(sub)}",
                    "mean_margin_B2": f"{np.mean(m2):.4f}" if m2 else "",
                    "worst_margin_B2": f"{min(m2):.4f}" if m2 else "",
                    "mean_rho": f"{np.mean(rh):.4f}" if rh else "",
                })
    write_csv(OUT_CSV_DIR / "covariance_by_depth_summary.csv", summary)

    for pooling in ("mean", "last"):
        print(f"\n{'='*100}\n{pooling.upper()} POOLING — tier structure by normalised depth\n{'='*100}")
        print(f"{'estimation set':28s}{'N':>6s}" +
              "".join(f"{'ND=%.1f' % d:>26s}" for d in DEPTHS))
        print(f"{'':34s}" + "".join(f"{'B1   B2   B3   mB2':>26s}" for _ in DEPTHS))
        for name in names:
            cells = []
            n_cov = ""
            for d in DEPTHS:
                s = [x for x in summary if x["pooling"] == pooling
                     and x["target_nd"] == d and x["estimation_set"] == name]
                if not s:
                    cells.append(f"{'':>26s}"); continue
                s = s[0]; n_cov = s["n_cov_samples"]
                cells.append(f"{s['B1']:>6s}{s['B2']:>6s}{s['B3']:>6s}"
                             f"{s['mean_margin_B2']:>8s}")
            print(f"{name:28s}{n_cov:>6}" + "".join(cells))



if __name__ == "__main__":
    np.random.seed(SEED)
    main()
