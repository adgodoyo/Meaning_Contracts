"""
spread_centering_eigenvalue.py -- two estimator checks for Appendix H.

  1. Centering 16 points leaves one structurally near-zero eigenvalue. The
     S0->S4 refinement-minus-control log-volume contrast is computed with and
     without it, on all 20 chains per arm, all four models, normalised depth
     0.5, and the shift reported as a percentage.
  2. Synthetic clouds that differ in shape (not only scale) test whether the
     n=16 estimators preserve the ordering of clouds by true effective
     dimensionality.

Sign convention as in Tables 14-16: positive means refinement is more contractive.

Outputs
-------
  results/robustness/spread_centering_eigenvalue.csv
  results/robustness/spread_shape_calibration.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FINAL_CODEBASE, MODELS, OUT_CSV_DIR, SEED,
    build_constraints_activation_index, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    load_activations, load_constraints_labels, log_volume, participation_ratio,
    normalised_depth_array,
)

TARGET_ND = 0.5
N_REAL = 16
N_TRIALS = 400


# ---------------------------------------------------------------------------
# Gap 1 — near-zero-eigenvalue cancellation, all chains, all models
# ---------------------------------------------------------------------------

def nearzero_cancellation() -> list[dict]:
    clab = load_constraints_labels()
    rows: list[dict] = []
    for model in MODELS:
        acts = load_activations(model, "constraints", "mean")
        X_all = acts[build_constraints_activation_index(model)]
        L = X_all.shape[1]
        layer = int(np.argmin(np.abs(normalised_depth_array(L) - TARGET_ND)))
        X_layer = X_all[:, layer, :].astype(np.float64)

        def eigs(chain: str, level: int) -> np.ndarray:
            m = (clab["chain_id"] == chain) & (clab["level"] == level)
            Xc = X_layer[m] - X_layer[m].mean(axis=0)
            s = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
            return (s ** 2) / max(m.sum() - 1, 1)

        def arm_gap(arm: str, drop_last: bool) -> float:
            chains = np.unique(clab["chain_id"][clab["arm"] == arm])
            assert len(chains) == 20, (arm, len(chains))
            vals = []
            for ch in chains:                      # ALL 20 chains, not the first 10
                e0, e4 = eigs(ch, 0), eigs(ch, 4)
                if drop_last:
                    e0, e4 = e0[:-1], e4[:-1]
                vals.append(log_volume(e4, rank=128) - log_volume(e0, rank=128))
            return float(np.mean(vals))

        out = {}
        for drop in (False, True):
            # paper convention: control minus refinement, so positive = refinement
            # occupies the smaller neighbourhood (is more contractive)
            out[drop] = arm_gap("commonality", drop) - arm_gap("refinement", drop)

        shift = abs(out[True] - out[False])
        pct = 100 * shift / max(abs(out[False]), 1e-12)
        n_min = min(len(eigs(c, l))
                    for c in np.unique(clab["chain_id"])[:6] for l in (0, 4))
        rows.append({
            "model": model, "layer": layer, "n_per_cloud": n_min,
            "n_chains_per_arm": 20,
            "contrast_paper_sign_with_nearzero_eig": f"{out[False]:.4f}",
            "contrast_paper_sign_without_nearzero_eig": f"{out[True]:.4f}",
            "abs_shift": f"{shift:.4f}",
            "relative_shift_pct": f"{pct:.2f}",
            "sign_preserved": bool(np.sign(out[True]) == np.sign(out[False])),
        })
        print(f"  {model:12s} L{layer:2d}  contrast {out[False]:+.4f} -> {out[True]:+.4f}  "
              f"shift {pct:.2f}%  sign preserved {np.sign(out[True]) == np.sign(out[False])}",
              flush=True)
        del acts, X_all, X_layer
    return rows


# ---------------------------------------------------------------------------
# Gap 2 — does the n=16 estimator rank true volume when SHAPE also varies?
# ---------------------------------------------------------------------------

def shape_calibration() -> list[dict]:
    """
    Clouds live in a d_eff-dimensional subspace of R^D. Spectral shape is set by
    an eigenvalue profile lam_i ~ i^-alpha: alpha=0 flat, alpha=1 moderate,
    alpha=2 steep. Within each shape, the profile is rescaled so the true
    log-volume over the subspace hits a chosen target, which lets shape and
    volume be varied independently.
    """
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    d_eff = 40
    idx = np.arange(1, d_eff + 1, dtype=np.float64)

    for D in (2048, 3072):
        for alpha, shape_name in ((0.0, "flat"), (1.0, "moderate"), (2.0, "steep")):
            base = idx ** (-alpha)
            for target_lv, vol_name in ((-172.694, "reference"),
                                        (-207.351, "small"),
                                        (-138.037, "large")):
                # rescale so sum(log(lam)) == target_lv over the d_eff subspace
                scale = np.exp((target_lv - np.sum(np.log(base))) / d_eff)
                lam = base * scale
                true_lv = float(np.sum(np.log(lam)))
                true_pr = float((lam.sum() ** 2) / (lam ** 2).sum())

                est_lv, est_pr = [], []
                for _ in range(N_TRIALS):
                    Z = rng.standard_normal((N_REAL, d_eff)) * np.sqrt(lam)
                    Q = rng.standard_normal((d_eff, D))
                    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
                    X = Z @ Q
                    Xc = X - X.mean(axis=0)
                    s = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
                    e = (s ** 2) / (N_REAL - 1)
                    est_lv.append(log_volume(e, rank=128))
                    est_pr.append(participation_ratio(e))
                rows.append({
                    "D": D, "d_eff": d_eff, "n": N_REAL, "n_trials": N_TRIALS,
                    "spectral_shape": shape_name, "alpha": alpha,
                    "volume_condition": vol_name,
                    "true_log_volume": f"{true_lv:.4f}",
                    "true_pr": f"{true_pr:.4f}",
                    "est_log_volume_mean": f"{np.mean(est_lv):.4f}",
                    "est_log_volume_sd": f"{np.std(est_lv):.4f}",
                    "est_pr_mean": f"{np.mean(est_pr):.4f}",
                    "est_pr_sd": f"{np.std(est_pr):.4f}",
                })
    return rows


def main() -> None:
    print("Gap 1 — near-zero-eigenvalue cancellation, all 20 chains per arm, all models")
    print("        (paper sign convention: positive = refinement more contractive)")
    nz = nearzero_cancellation()
    write_csv(OUT_CSV_DIR / "spread_centering_eigenvalue.csv", nz)

    print("\nGap 2 — n=16 estimator when spectral SHAPE varies, not only scale")
    sc = shape_calibration()
    write_csv(OUT_CSV_DIR / "spread_shape_calibration.csv", sc)

    print("\n  (a) volume ranking WITHIN each fixed shape:")
    ok_within = True
    for D in (2048, 3072):
        for shape in ("flat", "moderate", "steep"):
            sub = [r for r in sc if r["D"] == D and r["spectral_shape"] == shape]
            order_true = [r["volume_condition"] for r in
                          sorted(sub, key=lambda r: float(r["true_log_volume"]))]
            order_est = [r["volume_condition"] for r in
                         sorted(sub, key=lambda r: float(r["est_log_volume_mean"]))]
            good = order_true == order_est
            ok_within &= good
            print(f"    D={D} {shape:9s} true {order_true} est {order_est}  {'OK' if good else 'MISMATCH'}")

    print("\n  (b) volume ranking ACROSS shapes at matched true volume "
          "(the case differencing does not cover):")
    ok_across = True
    for D in (2048, 3072):
        for vol in ("small", "reference", "large"):
            sub = [r for r in sc if r["D"] == D and r["volume_condition"] == vol]
            tv = {r["spectral_shape"]: float(r["true_log_volume"]) for r in sub}
            ev = {r["spectral_shape"]: float(r["est_log_volume_mean"]) for r in sub}
            spread = max(ev.values()) - min(ev.values())
            print(f"    D={D} vol={vol:9s} true log-vol identical ({list(tv.values())[0]:.2f}) "
                  f"but estimates span {spread:.2f} nats: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in ev.items()))

    print("\n  (c) does shape variation ever invert a real volume difference?")
    for D in (2048, 3072):
        rows_D = [r for r in sc if r["D"] == D]
        inversions = 0
        total = 0
        for a in rows_D:
            for b in rows_D:
                ta, tb = float(a["true_log_volume"]), float(b["true_log_volume"])
                if ta >= tb - 1e-9:
                    continue
                total += 1
                if float(a["est_log_volume_mean"]) >= float(b["est_log_volume_mean"]):
                    inversions += 1
        print(f"    D={D}: {inversions} inversions out of {total} ordered pairs with a "
              f"genuine true-volume difference")
        ok_across &= inversions == 0

    print("\n  (d) participation ratio across shapes (scale-invariant, so this is the "
          "only axis PR can respond to):")
    for D in (2048,):
        for shape in ("flat", "moderate", "steep"):
            sub = [r for r in sc if r["D"] == D and r["spectral_shape"] == shape][0]
            print(f"    {shape:9s} true PR {float(sub['true_pr']):6.2f}  "
                  f"est PR {float(sub['est_pr_mean']):6.2f} "
                  f"(sd {float(sub['est_pr_sd']):.2f})")

    pcts = [float(r["relative_shift_pct"]) for r in nz]


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
