"""
12_adversarial_tables.py — Adversarial controls: PR and log-volume tables with CIs.

Produces two appendix tables for the paper:
  tab:adv_pr_ci   → adv_pr_ci.csv    (PR gap ± 95% CI at each arm's peak layer)
  tab:adv_lv_ci   → adv_lv_ci.csv   (log-volume gap ± 95% CI at log-vol peak layer)

Data sources
------------
  PR  : results/caches/refinement/mean/B1_pr_cache/{model}_{arm}_raw.npz
  LV  : results/caches/refinement/mean/B2/{model}.npz  (ref + com)
        results/caches/refinement/mean/ZY/{model}_{arm}_mean_id_lv.npz  (adv arms,
        computed here if missing and activations are available)

Notes
-----
  Refinement and commonality have n=20 chains; adversarial arms have n=4 chains.
  With n=4 the 95% CIs are very wide and often contain zero even when the
  direction is consistent. We report them for completeness but interpret only
  the central values.

Usage
-----
  python analysis/12_adversarial_tables.py
  python analysis/12_adversarial_tables.py --force   # recompute ZY lv cache
"""

import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from utils import (
    MODELS, TABLES_DIR, SEED,
    get_cache_dir,
    load_activations, load_constraints_labels,
    log_volume, peak_layer,
    ARM_REFINEMENT, ARM_COMMONALITY, ARM_NEAR_SYNONYM, ARM_SCOPE_HEDGE,
)

try:
    from sklearn.utils.extmath import randomized_svd
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

np.random.seed(SEED)

Z95       = 1.96
K_DEFLATE = 5
POOLING   = "mean"

ADV_ARMS  = [ARM_NEAR_SYNONYM, ARM_SCOPE_HEDGE]
ALL_ARMS  = [ARM_REFINEMENT, ARM_COMMONALITY] + ADV_ARMS

MODEL_LABELS = {
    "pythia-2.8b": "Pythia 2.8B",
    "llama3.2-3b": "Llama 3.2 3B",
    "qwen2.5-3b":  "Qwen 2.5 3B",
    "gemma4-e4b":  "Gemma 4 E4B",
}

ARM_LABELS = {
    ARM_REFINEMENT:  "Refinement",
    ARM_COMMONALITY: "Commonality",
    ARM_NEAR_SYNONYM: "Near-synonym",
    ARM_SCOPE_HEDGE:  "Scope-hedge",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _b1_cache(pooling: str = POOLING) -> Path:
    return get_cache_dir("refinement", pooling) / "B1_pr_cache"


def _b2_cache(pooling: str = POOLING) -> Path:
    return get_cache_dir("refinement", pooling) / "B2"


def _zy_cache(pooling: str = POOLING) -> Path:
    return get_cache_dir("refinement", pooling) / "ZY"


def _load(p: Path) -> dict:
    return dict(np.load(str(p), allow_pickle=True))


def _gap_ci(chain: np.ndarray, layer: int) -> tuple[float, float]:
    """chain: (n_chains, L, 5). Returns (mean_gap, half_ci) for S4-S0 at layer."""
    per_chain = chain[:, layer, 4] - chain[:, layer, 0]
    n = per_chain.shape[0]
    mean = float(per_chain.mean())
    ci   = float(Z95 * per_chain.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, ci


# ── PR peak-layer table ───────────────────────────────────────────────────────

def build_pr_table() -> list[dict]:
    """
    Each arm's peak layer is the argmax of the smoothed mean gap profile
    (same as 04_participation_ratio.py: peak_layer with smooth=3).
    Gap = PR(S4) - PR(S0).
    """
    rows = []
    b1 = _b1_cache()
    for model in MODELS:
        row = {"model": model}
        for arm in ALL_ARMS:
            p = b1 / f"{model}_{arm}_raw.npz"
            if not p.exists():
                row[arm] = ("n/a", "n/a", "n/a", "n/a")
                continue
            d = _load(p)
            chain = d["chain_pr"]                                        # (n, L, 5)
            mean_gap_profile = (chain[:, :, 4] - chain[:, :, 0]).mean(axis=0)  # (L,)
            lstar = peak_layer(mean_gap_profile, smooth=3)
            val, ci = _gap_ci(chain, lstar)
            n = chain.shape[0]
            row[arm] = (val, ci, n, lstar)
        rows.append(row)
    return rows


# ── Log-volume: load or compute ───────────────────────────────────────────────

def _precompute_deflation(acts: np.ndarray, mask_arm: np.ndarray,
                          seed: int = SEED):
    L, D = acts.shape[1], acts.shape[2]
    arm_idx = np.where(mask_arm)[0]
    N_arm = len(arm_idx)
    arm_mean = np.zeros((L, D), dtype=np.float64)
    Vt_top5  = np.zeros((L, K_DEFLATE, D), dtype=np.float64)
    for l in range(L):
        X_l = acts[arm_idx, l, :].astype(np.float64)
        arm_mean[l] = X_l.mean(axis=0)
        Xc_l = X_l - arm_mean[l]
        k_eff = min(K_DEFLATE, N_arm - 1, D)
        _, _, Vt = randomized_svd(Xc_l, n_components=k_eff, random_state=seed)
        Vt_top5[l, :k_eff] = Vt
    return arm_mean, Vt_top5


def _apply_top5(X_ch: np.ndarray, arm_mean: np.ndarray, Vt: np.ndarray):
    Xc   = X_ch - arm_mean[:, None, :]
    proj = np.einsum("lnd,lkd->lnk", Xc, Vt)
    recon = np.einsum("lnk,lkd->lnd", proj, Vt)
    return Xc - recon


def _compute_adv_lv(model: str, arm: str, force: bool = False) -> np.ndarray:
    """Returns chain_lv (n_chains, L, 5) for an adversarial arm."""
    tag = "mean"
    cache_path = _zy_cache() / f"{model}_{arm}_{tag}_id_lv.npz"
    if cache_path.exists() and not force:
        d = np.load(str(cache_path))
        print(f"  [{model}] {arm} lv loaded from ZY cache", flush=True)
        return d["chain_lv"]

    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn not available; cannot compute deflated lv.")

    print(f"  [{model}] Computing top5-deflated lv for {arm}...", flush=True)
    acts = load_activations(model, "constraints", POOLING)
    clab = load_constraints_labels()
    # Activations may include extra rows (extra arms not in the labels file).
    # Slice to match the labels which cover only the 4 arms used in the analysis.
    acts = acts[:len(clab["arm"])]
    N, L, D = acts.shape

    mask_arm = clab["arm"] == arm
    chains   = np.unique(clab["chain_id"][mask_arm])
    n_chains = len(chains)

    arm_mean, Vt_top5 = _precompute_deflation(acts, mask_arm)

    chain_lv = np.full((n_chains, L, 5), np.nan)

    for li in range(5):
        mask_lv = mask_arm & (clab["level"] == li)
        print(f"    level {li}/4", flush=True)
        for ci, ch in enumerate(chains):
            mask_ch = mask_lv & (clab["chain_id"] == ch)
            if mask_ch.sum() < 3:
                continue
            X_ch  = acts[mask_ch].astype(np.float64).transpose(1, 0, 2)  # (L,n,D)
            n_sent = X_ch.shape[1]
            X_def = _apply_top5(X_ch, arm_mean, Vt_top5)
            Xc    = X_def - X_def.mean(axis=1, keepdims=True)
            S     = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
            eigs  = (S ** 2) / max(n_sent - 1, 1)
            for l in range(L):
                chain_lv[ci, l, li] = log_volume(eigs[l])

    np.savez(str(cache_path), chain_lv=chain_lv,
             chain_id=np.full_like(chain_lv, np.nan))
    print(f"  [{model}] {arm} lv cached → ZY", flush=True)
    return chain_lv


def build_lv_table(force: bool = False) -> list[dict]:
    """
    Peak layer = layer with most-negative mean lv gap (S4−S0) for each arm.
    Ref and com: from B2 chain_lv (n=20).
    Adv arms:    computed via _compute_adv_lv (n=4).
    """
    rows = []
    b2   = _b2_cache()
    for model in MODELS:
        row = {"model": model}
        b2d = _load(b2 / f"{model}.npz")

        # Refinement
        ref_chain = b2d["ref_chain_lv"]  # (20, L, 5)
        ref_gap_profile = (ref_chain[:, :, 4] - ref_chain[:, :, 0]).mean(axis=0)
        ref_peak = peak_layer(ref_gap_profile, smooth=3)
        val, ci  = _gap_ci(ref_chain, ref_peak)
        row[ARM_REFINEMENT] = (val, ci, ref_chain.shape[0], ref_peak)

        # Commonality
        com_chain = b2d["com_chain_lv"]  # (20, L, 5)
        com_gap_profile = (com_chain[:, :, 4] - com_chain[:, :, 0]).mean(axis=0)
        com_peak = peak_layer(com_gap_profile, smooth=3)
        val, ci  = _gap_ci(com_chain, com_peak)
        row[ARM_COMMONALITY] = (val, ci, com_chain.shape[0], com_peak)

        # Adversarial
        for arm in ADV_ARMS:
            try:
                chain_lv = _compute_adv_lv(model, arm, force=force)
                gap_prof = (chain_lv[:, :, 4] - chain_lv[:, :, 0]).mean(axis=0)
                lstar = peak_layer(gap_prof, smooth=3)
                val, ci = _gap_ci(chain_lv, lstar)
                n = chain_lv.shape[0]
                row[arm] = (val, ci, n, lstar)
            except Exception as e:
                print(f"  [{model}] {arm} lv failed: {e}", flush=True)
                row[arm] = ("n/a", "n/a", "n/a", "n/a")

        rows.append(row)
    return rows


# ── CSV saving ────────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], filename: str, arms: list[str]) -> None:
    path = TABLES_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model"]
        for arm in arms:
            header += [f"{arm}_gap", f"{arm}_ci95", f"{arm}_n", f"{arm}_peak_layer"]
        w.writerow(header)
        for row in rows:
            line = [row["model"]]
            for arm in arms:
                v = row.get(arm, ("n/a",) * 4)
                if v[0] == "n/a":
                    line += ["n/a", "n/a", "n/a", "n/a"]
                else:
                    line += [f"{v[0]:.4f}", f"{v[1]:.4f}", v[2], v[3]]
            w.writerow(line)
    print(f"  → {path.name}", flush=True)


# ── LaTeX printing ────────────────────────────────────────────────────────────

def _fmt(val, ci, n):
    if val == "n/a":
        return r"\text{n/a}"
    sign = "+" if val >= 0 else ""
    ci_note = (r"{\scriptsize\,\(\pm" + f"{ci:.3f}" + r"\)}") if not np.isnan(ci) else ""
    return f"{sign}{val:.3f}{ci_note}"


def print_latex_pr(rows: list[dict]) -> None:
    print("\n% === tab:adv_pr_ci ===")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Model & Refinement & Commonality & Near-synonym & Scope-hedge \\")
    print(r"\midrule")
    for row in rows:
        mlabel = MODEL_LABELS[row["model"]]
        cols = []
        for arm in ALL_ARMS:
            v = row.get(arm, ("n/a", float("nan"), "n/a", "n/a"))
            cols.append(_fmt(v[0], v[1] if not isinstance(v[1], str) else float("nan"), v[2]))
        print(f"{mlabel} & {' & '.join(cols)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_latex_lv(rows: list[dict]) -> None:
    print("\n% === tab:adv_lv_ci ===")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Model & Refinement & Commonality & Near-synonym & Scope-hedge \\")
    print(r"\midrule")
    for row in rows:
        mlabel = MODEL_LABELS[row["model"]]
        cols = []
        for arm in ALL_ARMS:
            v = row.get(arm, ("n/a", float("nan"), "n/a", "n/a"))
            ci_val = v[1] if not isinstance(v[1], str) else float("nan")
            cols.append(_fmt(v[0], ci_val, v[2]))
        print(f"{mlabel} & {' & '.join(cols)} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="12: Adversarial controls PR and log-volume tables with CIs")
    parser.add_argument("--force", action="store_true",
                        help="Recompute ZY log-volume cache for adversarial arms")
    args = parser.parse_args()

    print("Building PR table...", flush=True)
    pr_rows = build_pr_table()
    _save_csv(pr_rows, "adv_pr_ci.csv", ALL_ARMS)
    print_latex_pr(pr_rows)

    print("\nBuilding log-volume table...", flush=True)
    lv_rows = build_lv_table(force=args.force)
    _save_csv(lv_rows, "adv_lv_ci.csv", ALL_ARMS)
    print_latex_lv(lv_rows)

    print("\n12 complete.", flush=True)
