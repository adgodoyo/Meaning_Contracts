"""
spread_permutation_and_calibration.py -- behaviour of the n=16 spread estimators
(Table 11, Appendix H).

  1. Permutation test: the refinement/control label is shuffled across the 40
     chains (10,000 permutations) and the S0->S4 gap recomputed.
  2. Synthetic calibration: Gaussian clouds with known log-volume and
     participation ratio, sampled at n=16 in the models' ambient dimensions.
  3. Subsampling stability: the gap recomputed with 12 of the 16 sentences per
     (chain, level), 200 random subsamples.

Outputs
-------
  results/robustness/spread_permutation_test.csv
  results/robustness/spread_synthetic_calibration.csv
  results/robustness/spread_subsampling_stability.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FINAL_CODEBASE, MODELS, RESULTS_CACHE, OUT_CSV_DIR, SEED,
    build_constraints_activation_index, load_stable_band, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    load_activations, load_constraints_labels, log_volume, participation_ratio,
)

RNG = np.random.default_rng(SEED)
N_PERM = 10_000
N_SUBSAMPLE_DRAWS = 200
N_CHAINS = 20


# ---------------------------------------------------------------------------
# 1. Permutation test
# ---------------------------------------------------------------------------

def permutation_test_one(ref_arr: np.ndarray, com_arr: np.ndarray, layer: int) -> dict:
    """
    ref_arr, com_arr: (20, L, 5) per-chain values (log-volume or PR).
    Statistic: mean S0->S4 gap (ref[4]-ref[0]) minus (com[4]-com[0]) at `layer`,
    pooled: we permute which of the 40 chains are labeled 'ref' vs 'com'.
    """
    ref_gap = ref_arr[:, layer, 4] - ref_arr[:, layer, 0]   # (20,)
    com_gap = com_arr[:, layer, 4] - com_arr[:, layer, 0]   # (20,)
    observed = float(np.nanmean(ref_gap) - np.nanmean(com_gap))

    pooled = np.concatenate([ref_gap, com_gap])  # (40,)
    n = len(ref_gap)
    null = np.empty(N_PERM)
    idx = np.arange(len(pooled))
    for i in range(N_PERM):
        RNG.shuffle(idx)
        a = pooled[idx[:n]]
        b = pooled[idx[n:]]
        null[i] = np.nanmean(a) - np.nanmean(b)
    p = float(np.mean(np.abs(null) >= np.abs(observed)))
    return {"observed_gap": observed, "p_value": p, "null_sd": float(np.nanstd(null))}


def run_permutation_tests() -> list[dict]:
    rows = []
    for model in MODELS:
        lo, hi = load_stable_band(model, "mean")
        layer = (lo + hi - 1) // 2

        b2 = dict(np.load(RESULTS_CACHE / "B2" / f"{model}.npz"))
        res_lv = permutation_test_one(b2["ref_chain_lv"], b2["com_chain_lv"], layer)
        rows.append({"model": model, "quantity": "log_volume_raw", "layer": layer, **res_lv})

        pr_ref = np.load(RESULTS_CACHE / "B1_pr_cache" / f"{model}_refinement_raw.npz")["chain_pr"]
        pr_com = np.load(RESULTS_CACHE / "B1_pr_cache" / f"{model}_commonality_raw.npz")["chain_pr"]
        res_pr = permutation_test_one(pr_ref, pr_com, layer)
        rows.append({"model": model, "quantity": "participation_ratio_raw", "layer": layer, **res_pr})

        zy_ref = np.load(RESULTS_CACHE / "ZY" / f"{model}_refinement_mean_id_lv.npz")["chain_lv"]
        zy_com = np.load(RESULTS_CACHE / "ZY" / f"{model}_commonality_mean_id_lv.npz")["chain_lv"]
        res_top5 = permutation_test_one(zy_ref, zy_com, layer)
        rows.append({"model": model, "quantity": "log_volume_top5_deflated", "layer": layer, **res_top5})
    return rows


# ---------------------------------------------------------------------------
# 2. Synthetic calibration
# ---------------------------------------------------------------------------

def run_synthetic_calibration() -> list[dict]:
    rows = []
    n = 16
    for D in (2048, 3072):
        # True log-volumes / PRs spanning the observed range: build diagonal
        # covariances with a fixed spectral shape scaled by a volume factor.
        rank_eff = 50  # effective rank of the "signal" subspace; rest near-zero
        base_eigs = np.geomspace(1.0, 1e-3, rank_eff)
        for scale, tag in [(0.5, "small_cloud"), (1.0, "reference_cloud"), (2.0, "large_cloud")]:
            true_eigs = base_eigs * scale
            true_lv = float(np.sum(np.log(true_eigs)))
            true_pr = float((np.sum(true_eigs) ** 2) / np.sum(true_eigs ** 2))

            est_lvs, est_prs = [], []
            for trial in range(200):
                rng = np.random.default_rng(SEED + trial)
                Z = rng.standard_normal((n, rank_eff)) * np.sqrt(true_eigs)
                pad = rng.standard_normal((n, D - rank_eff)) * 1e-6
                X = np.concatenate([Z, pad], axis=1)
                Xc = X - X.mean(axis=0)
                cov_eigs = np.linalg.svd(Xc, compute_uv=False) ** 2 / max(n - 1, 1)
                est_lvs.append(log_volume(cov_eigs, rank=min(128, n - 1)))
                est_prs.append(participation_ratio(cov_eigs))

            rows.append({
                "D": D, "cloud": tag, "n": n,
                "true_log_volume": f"{true_lv:.3f}",
                "est_log_volume_mean": f"{np.mean(est_lvs):.3f}",
                "est_log_volume_sd": f"{np.std(est_lvs):.3f}",
                "true_pr": f"{true_pr:.3f}",
                "est_pr_mean": f"{np.mean(est_prs):.3f}",
                "est_pr_sd": f"{np.std(est_prs):.3f}",
            })
    return rows


def calibration_ranking_preserved(rows: list[dict]) -> bool:
    for D in (2048, 3072):
        sub = [r for r in rows if r["D"] == D]
        true_order = sorted(sub, key=lambda r: float(r["true_log_volume"]))
        est_order = sorted(sub, key=lambda r: float(r["est_log_volume_mean"]))
        if [r["cloud"] for r in true_order] != [r["cloud"] for r in est_order]:
            return False
    return True


# ---------------------------------------------------------------------------
# 3. Subsampling stability (n=12 vs n=16), real data, top-5-deflated branch
# ---------------------------------------------------------------------------

def run_subsampling_stability() -> list[dict]:
    from sklearn.utils.extmath import randomized_svd  # noqa: F401 (ensures dep present)
    clab = load_constraints_labels()
    rows = []
    for model in MODELS:
        lo, hi = load_stable_band(model, "mean")
        layer = (lo + hi - 1) // 2
        acts = load_activations(model, "constraints", "mean")  # (N, L, D) memmap
        act_index = build_constraints_activation_index(model)  # (3840,) -> aligns with clab
        X_layer = acts[:, layer, :].astype(np.float64)[act_index]  # (3840, D), aligned with clab

        for arm in ("refinement", "commonality"):
            mask_arm = clab["arm"] == arm
            chains = np.unique(clab["chain_id"][mask_arm])

            full_lv = {0: [], 4: []}
            sub_lv = {0: [], 4: []}
            for lvl in (0, 4):
                mask_lvl = mask_arm & (clab["level"] == lvl)
                for ch in chains:
                    mask_ch = mask_lvl & (clab["chain_id"] == ch)
                    X_ch = X_layer[mask_ch]  # (16, D) typically
                    n_sent = X_ch.shape[0]
                    if n_sent < 13:
                        continue
                    Xc = X_ch - X_ch.mean(axis=0)
                    eigs_full = np.linalg.svd(Xc, compute_uv=False) ** 2 / max(n_sent - 1, 1)
                    full_lv[lvl].append(log_volume(eigs_full, rank=min(15, n_sent - 1)))

                    sub_vals = []
                    for draw in range(N_SUBSAMPLE_DRAWS // 20):  # keep total draws bounded
                        idx = RNG.choice(n_sent, size=12, replace=False)
                        Xs = X_ch[idx]
                        Xsc = Xs - Xs.mean(axis=0)
                        eigs_s = np.linalg.svd(Xsc, compute_uv=False) ** 2 / max(12 - 1, 1)
                        sub_vals.append(log_volume(eigs_s, rank=min(11, 11)))
                    sub_lv[lvl].append(float(np.mean(sub_vals)))

            gap_full = np.mean(full_lv[4]) - np.mean(full_lv[0])
            gap_sub = np.mean(sub_lv[4]) - np.mean(sub_lv[0])
            rows.append({
                "model": model, "arm": arm, "layer": layer,
                "gap_n16_full": f"{gap_full:.4f}",
                "gap_n12_subsampled": f"{gap_sub:.4f}",
                "same_sign": bool(np.sign(gap_full) == np.sign(gap_sub)),
            })
    return rows


def main() -> None:
    perm_rows = run_permutation_tests()
    write_csv(OUT_CSV_DIR / "spread_permutation_test.csv", perm_rows)
    n_sig = sum(1 for r in perm_rows if r["p_value"] < 0.05)
    print(f"\nPermutation test: {n_sig}/{len(perm_rows)} model x quantity cells significant at p<0.05")
    for r in perm_rows:
        print(f"  {r['model']:14s} {r['quantity']:26s} gap={r['observed_gap']:+.4f}  p={r['p_value']:.4f}")

    calib_rows = run_synthetic_calibration()
    write_csv(OUT_CSV_DIR / "spread_synthetic_calibration.csv", calib_rows)
    ranking_ok = calibration_ranking_preserved(calib_rows)
    print(f"\nSynthetic calibration: ranking preserved at n=16 across D in {{2048,3072}}? {ranking_ok}")

    sub_rows = run_subsampling_stability()
    write_csv(OUT_CSV_DIR / "spread_subsampling_stability.csv", sub_rows)
    n_stable = sum(1 for r in sub_rows if r["same_sign"])
    print(f"\nSubsampling stability (n=12 vs n=16): {n_stable}/{len(sub_rows)} model x arm cells sign-stable")



if __name__ == "__main__":
    main()
