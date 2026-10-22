"""
03_logvolume_refinement.py — Z4 and Z5 figures with top-5 PC deflation applied first.

Before computing any metric, the top-5 global principal components of each arm
are projected out of every sentence's activation (per layer).  This removes the
dominant surface/template variance that nuisance-deflation analyses (A3) showed
is captured in the first few PCs.

Metrics:
  PR  — loaded directly from B1 cache top5 branch (already computed).
  ID  — computed on the fly: deflate → MLE intrinsic dimension.
  lv  — computed on the fly: deflate → log-volume.

ID and lv results are cached to results/cache[_last]/ZY/ so subsequent runs
are instant.

Figures produced (same set as Z4 + Z5):
  Z4-style:  profiles S0-S2, S3-S4, gap_from_S0, gap_from_S1, gap_late_S2, gap_S3_S4
  Z5-style:  span1, span2, span3, span4

Usage
-----
  python analysis/03_logvolume_refinement.py
  python analysis/03_logvolume_refinement.py --models qwen2.5-3b
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, ARM_COLORS,
    get_cache_dir, get_figures_dir,
    load_activations, load_constraints_labels,
    mle_intrinsic_dim, log_volume,
    normalised_depth_array,
    save_figure, setup_plt, SEED,
)

setup_plt()
np.random.seed(SEED)


def _get_zy_cache(pooling: str) -> Path:
    d = get_cache_dir("refinement", pooling) / "ZY"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _get_b1_cache(pooling: str) -> Path:
    return get_cache_dir("refinement", pooling) / "B1_pr_cache"


TECHNIQUES  = ["pr", "id", "lv"]
TECH_LABELS = {"pr": "Participation Ratio", "id": "Intrinsic Dimension", "lv": "Log-volume"}
TECH_UNITS  = {"pr": "PR", "id": "MLE ID", "lv": "log-vol"}

N_CHAINS = 20
Z_95     = 1.96
K_DEFLATE = 5


def _level_colors():
    cmap  = mcm.get_cmap("viridis")
    stops = np.linspace(0.1, 0.9, 5)
    return [matplotlib.colors.to_hex(cmap(t)) for t in stops]

LEVEL_COLORS = _level_colors()
_COL_REF = ARM_COLORS["refinement"]
_COL_COM = ARM_COLORS["commonality"]


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig, name: str, model: str, pooling: str = "mean") -> None:
    save_figure(fig, pooling, "refinement_logvol", f"{model}_{name}")


# ── PR loader (top5 branch already cached) ────────────────────────────────────

def _load_pr_top5(model: str, arm: str, pooling: str = "mean"):
    """Return chain_pr (20, L, 5) from B1 top5 cache, or None."""
    p = _get_b1_cache(pooling) / f"{model}_{arm}_top5.npz"
    if not p.exists():
        return None
    d = np.load(str(p))
    return d["chain_pr"] if "chain_pr" in d else None


# ── ID / lv with top5 deflation (computed on the fly, cached) ─────────────────

def _precompute_deflation(acts: np.ndarray, mask_arm: np.ndarray):
    """
    Compute arm_mean (L, D) and Vt_top5 (L, K, D) for the given arm mask.
    Uses randomized SVD — same logic as B1.
    """
    L, D    = acts.shape[1], acts.shape[2]
    arm_idx = np.where(mask_arm)[0]
    N_arm   = len(arm_idx)

    arm_mean = np.zeros((L, D), dtype=np.float64)
    Vt_top5  = np.zeros((L, K_DEFLATE, D), dtype=np.float64)

    for l in range(L):
        X_l = acts[arm_idx, l, :].astype(np.float64)
        arm_mean[l] = X_l.mean(axis=0)
        Xc_l = X_l - arm_mean[l]
        k_eff = min(K_DEFLATE, N_arm - 1, D)
        _, _, Vt = randomized_svd(Xc_l, n_components=k_eff, random_state=SEED)
        Vt_top5[l, :k_eff] = Vt

    return arm_mean, Vt_top5


def _apply_top5(X_ch: np.ndarray, arm_mean: np.ndarray, Vt: np.ndarray):
    """
    X_ch : (L, n, D)  raw activations
    Returns (L, n, D) top-5 deflated activations.
    """
    Xc   = X_ch - arm_mean[:, None, :]             # (L, n, D)
    proj = np.einsum("lnd,lkd->lnk", Xc, Vt)       # (L, n, k)
    recon = np.einsum("lnk,lkd->lnd", proj, Vt)    # (L, n, D)
    return Xc - recon


def _compute_id_lv_top5(model: str, arm: str, pooling: str):
    """
    Compute top5-deflated ID and log-vol per chain per level per layer.
    Returns (chain_id, chain_lv) each (n_chains, L, 5).
    Caches result to _get_zy_cache(pooling).
    """
    tag   = "last" if pooling == "last" else "mean"
    cache_path = _get_zy_cache(pooling) / f"{model}_{arm}_{tag}_id_lv.npz"

    if cache_path.exists():
        d = np.load(str(cache_path))
        print(f"  [{model}] {arm} ID/lv top5 loaded from ZY cache", flush=True)
        return d["chain_id"], d["chain_lv"]

    print(f"  [{model}] Computing top5-deflated ID/lv for {arm}...", flush=True)
    acts  = load_activations(model, "constraints", pooling)  # (N, L, D)
    clab  = load_constraints_labels()
    N, L, D = acts.shape

    mask_arm = clab["arm"] == arm
    chains   = np.unique(clab["chain_id"][mask_arm])
    n_chains = len(chains)

    arm_mean, Vt_top5 = _precompute_deflation(acts, mask_arm)

    chain_id = np.full((n_chains, L, 5), np.nan)
    chain_lv = np.full((n_chains, L, 5), np.nan)

    for li in range(5):
        mask_lv = mask_arm & (clab["level"] == li)
        print(f"    level {li}/4", flush=True)

        for ci, ch in enumerate(chains):
            mask_ch = mask_lv & (clab["chain_id"] == ch)
            if mask_ch.sum() < 3:
                continue

            X_ch   = acts[mask_ch].astype(np.float64).transpose(1, 0, 2)  # (L,n,D)
            n_sent = X_ch.shape[1]
            X_def  = _apply_top5(X_ch, arm_mean, Vt_top5)                  # (L,n,D)

            Xc       = X_def - X_def.mean(axis=1, keepdims=True)
            S        = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
            eigs_all = (S ** 2) / max(n_sent - 1, 1)                       # (L, n)

            for l in range(L):
                X_sent = X_def[l].T.copy()          # (D, n) → transpose for mle
                chain_id[ci, l, li] = mle_intrinsic_dim(
                    X_def[l], k=min(5, n_sent - 1))
                chain_lv[ci, l, li] = log_volume(eigs_all[l])

    np.savez(str(cache_path), chain_id=chain_id, chain_lv=chain_lv)
    print(f"  [{model}] {arm} ID/lv top5 cached → ZY cache", flush=True)
    return chain_id, chain_lv


# ── Data assembly ─────────────────────────────────────────────────────────────

def _get_data(model: str, tech: str, arm: str, pooling: str):
    """Return (chain_arr (20, L, 5), mean_arr (L,5), sd_arr (L,5)) or None."""
    if tech == "pr":
        chain = _load_pr_top5(model, arm, pooling)
        if chain is None:
            return None
    else:
        tag = "last" if pooling == "last" else "mean"
        id_chain, lv_chain = _compute_id_lv_top5(model, arm, pooling)
        chain = id_chain if tech == "id" else lv_chain

    mean_arr = np.nanmean(chain, axis=0)
    sd_arr   = np.nanstd( chain, axis=0, ddof=1)
    return chain, mean_arr, sd_arr


# ── CI helpers ────────────────────────────────────────────────────────────────

def _gap_exact(chain_hi, chain_lo):
    gaps = chain_hi - chain_lo
    mean = np.nanmean(gaps, axis=0)
    ci   = Z_95 * np.nanstd(gaps, axis=0, ddof=1) / np.sqrt(N_CHAINS)
    return mean, ci


def _resolve_gap(hi, lo, ref_chain, ref_mean, ref_sd, com_chain, com_mean, com_sd):
    if ref_chain is not None:
        rg, rc = _gap_exact(ref_chain[:, :, hi], ref_chain[:, :, lo])
        cg, cc = _gap_exact(com_chain[:, :, hi], com_chain[:, :, lo])
        lbl = "exact 95% CI"
    else:
        rg = ref_mean[:, hi] - ref_mean[:, lo]
        rc = Z_95 * np.sqrt(ref_sd[:, hi]**2 + ref_sd[:, lo]**2) / np.sqrt(N_CHAINS)
        cg = com_mean[:, hi] - com_mean[:, lo]
        cc = Z_95 * np.sqrt(com_sd[:, hi]**2 + com_sd[:, lo]**2) / np.sqrt(N_CHAINS)
        lbl = "conservative 95% CI"
    return rg, rc, cg, cc, lbl


def _span_exact(chain, span):
    pairs = [(lo, lo + span) for lo in range(5 - span)]
    per_chain = np.mean([chain[:, :, hi] - chain[:, :, lo]
                         for lo, hi in pairs], axis=0)
    mean = np.nanmean(per_chain, axis=0)
    ci   = Z_95 * np.nanstd(per_chain, axis=0, ddof=1) / np.sqrt(N_CHAINS)
    return mean, ci


# ── Z4-style figures ──────────────────────────────────────────────────────────

def _fig_profiles(nd, ref_chain, ref_mean, ref_sd, com_chain, com_mean, com_sd,
                  levels, tech, model, suffix, pooling: str):
    ref_ci = Z_95 * ref_sd / np.sqrt(N_CHAINS)
    com_ci = Z_95 * com_sd / np.sqrt(N_CHAINS)
    level_str = f"S{levels[0]}–S{levels[-1]}"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        f"{model}  —  {TECH_LABELS[tech]} [top-5 deflated]: levels {level_str}\n"
        f"(refinement vs control, exact 95% CI)",
        fontsize=10,
    )
    for k in levels:
        col = LEVEL_COLORS[k]
        ax.plot(nd, ref_mean[:, k], color=col, linestyle="-",  linewidth=2.0)
        ax.fill_between(nd, ref_mean[:, k] - ref_ci[:, k],
                            ref_mean[:, k] + ref_ci[:, k], color=col, alpha=0.15)
        ax.plot(nd, com_mean[:, k], color=col, linestyle="--", linewidth=2.0)
        ax.fill_between(nd, com_mean[:, k] - com_ci[:, k],
                            com_mean[:, k] + com_ci[:, k], color=col, alpha=0.08)
    ax.set_xlabel("Normalised depth  ℓ / (L−1)")
    ax.set_ylabel(TECH_UNITS[tech])
    level_handles = [plt.Line2D([0],[0], color=LEVEL_COLORS[k], linewidth=2,
                                label=f"S{k}") for k in levels]
    style_handles = [
        plt.Line2D([0],[0], color="dimgray", linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0],[0], color="dimgray", linestyle="--", linewidth=2,
                   label="Control"),
        matplotlib.patches.Patch(facecolor="dimgray", alpha=0.20, label="95% CI"),
    ]
    ax.legend(handles=level_handles + style_handles, fontsize=8, ncol=4, loc="best")
    _save(fig, f"ZY_{tech}_{suffix}", model, pooling)


def _fig_gap_from_anchor(nd, ref_chain, ref_mean, ref_sd,
                          com_chain, com_mean, com_sd, anchor, tech, model, pooling: str):
    ks = list(range(anchor + 1, 5))
    ci_labels = set()
    fig, ax = plt.subplots(figsize=(9, 5))

    for k in ks:
        col = LEVEL_COLORS[k]
        rg, rc, cg, cc, lbl = _resolve_gap(
            k, anchor, ref_chain, ref_mean, ref_sd, com_chain, com_mean, com_sd)
        ci_labels.add(lbl)
        ax.plot(nd, rg, color=col, linestyle="-",  linewidth=2.0)
        ax.fill_between(nd, rg - rc, rg + rc, color=col, alpha=0.15)
        ax.plot(nd, cg, color=col, linestyle="--", linewidth=2.0)
        ax.fill_between(nd, cg - cc, cg + cc, color=col, alpha=0.08)

    ci_label = list(ci_labels)[0]
    ax.set_title(
        f"{model}  —  {TECH_LABELS[tech]} [top-5 deflated]: gaps from S{anchor}\n"
        f"(refinement vs control, {ci_label})",
        fontsize=10,
    )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel("Normalised depth  ℓ / (L−1)")
    ax.set_ylabel(f"Δ {TECH_UNITS[tech]}")
    level_handles = [plt.Line2D([0],[0], color=LEVEL_COLORS[k], linewidth=2,
                                label=f"S{anchor}→S{k}") for k in ks]
    style_handles = [
        plt.Line2D([0],[0], color="dimgray", linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0],[0], color="dimgray", linestyle="--", linewidth=2,
                   label="Control"),
        matplotlib.patches.Patch(facecolor="dimgray", alpha=0.20, label=ci_label),
    ]
    ax.legend(handles=level_handles + style_handles, fontsize=8, ncol=4, loc="best")
    _save(fig, f"ZY_{tech}_gap_from_S{anchor}", model, pooling)


def _fig_gap_late_S2(nd, ref_chain, ref_mean, ref_sd,
                      com_chain, com_mean, com_sd, tech, model, pooling: str):
    pairs       = [(2, 3), (2, 4)]
    pair_colors = [LEVEL_COLORS[3], LEVEL_COLORS[4]]
    ci_labels   = set()
    fig, ax = plt.subplots(figsize=(9, 5))

    for (lo, hi), col in zip(pairs, pair_colors):
        rg, rc, cg, cc, lbl = _resolve_gap(
            hi, lo, ref_chain, ref_mean, ref_sd, com_chain, com_mean, com_sd)
        ci_labels.add(lbl)
        ax.plot(nd, rg, color=col, linestyle="-",  linewidth=2.0)
        ax.fill_between(nd, rg - rc, rg + rc, color=col, alpha=0.15)
        ax.plot(nd, cg, color=col, linestyle="--", linewidth=2.0)
        ax.fill_between(nd, cg - cc, cg + cc, color=col, alpha=0.08)

    ci_label = list(ci_labels)[0]
    ax.set_title(
        f"{model}  —  {TECH_LABELS[tech]} [top-5 deflated]: late gaps S2→S3, S2→S4\n"
        f"(refinement vs control, {ci_label})",
        fontsize=10,
    )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel("Normalised depth  ℓ / (L−1)")
    ax.set_ylabel(f"Δ {TECH_UNITS[tech]}")
    gap_handles = [
        plt.Line2D([0],[0], color=pair_colors[0], linewidth=2, label="S2→S3"),
        plt.Line2D([0],[0], color=pair_colors[1], linewidth=2, label="S2→S4"),
    ]
    style_handles = [
        plt.Line2D([0],[0], color="dimgray", linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0],[0], color="dimgray", linestyle="--", linewidth=2,
                   label="Control"),
        matplotlib.patches.Patch(facecolor="dimgray", alpha=0.20, label=ci_label),
    ]
    ax.legend(handles=gap_handles + style_handles, fontsize=8, ncol=3, loc="best")
    _save(fig, f"ZY_{tech}_gap_late_S2", model, pooling)


def _fig_gap_S3_S4(nd, ref_chain, ref_mean, ref_sd,
                    com_chain, com_mean, com_sd, tech, model, pooling: str):
    rg, rc, cg, cc, ci_label = _resolve_gap(
        4, 3, ref_chain, ref_mean, ref_sd, com_chain, com_mean, com_sd)
    col = LEVEL_COLORS[4]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        f"{model}  —  {TECH_LABELS[tech]} [top-5 deflated]: gap S3→S4\n"
        f"(refinement vs control, {ci_label})",
        fontsize=10,
    )
    ax.plot(nd, rg, color=col,      linestyle="-",  linewidth=2.0)
    ax.fill_between(nd, rg - rc, rg + rc, color=col,      alpha=0.15)
    ax.plot(nd, cg, color=_COL_COM, linestyle="--", linewidth=2.0)
    ax.fill_between(nd, cg - cc, cg + cc, color=_COL_COM, alpha=0.15)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel("Normalised depth  ℓ / (L−1)")
    ax.set_ylabel(f"Δ {TECH_UNITS[tech]}")
    ax.legend(handles=[
        plt.Line2D([0],[0], color=col,      linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0],[0], color=_COL_COM, linestyle="--", linewidth=2,
                   label="Control"),
        matplotlib.patches.Patch(facecolor="dimgray", alpha=0.20, label=ci_label),
    ], fontsize=8, loc="best")
    _save(fig, f"ZY_{tech}_gap_S3_S4", model, pooling)


# ── Z5-style figures ──────────────────────────────────────────────────────────

def _fig_span(nd, ref_chain, com_chain, span, tech, model, pooling: str):
    pairs    = [(lo, lo + span) for lo in range(5 - span)]
    pair_str = ", ".join(f"S{lo}→S{hi}" for lo, hi in pairs)

    ref_gap, ref_ci = _span_exact(ref_chain, span)
    com_gap, com_ci = _span_exact(com_chain, span)

    col_ref = ARM_COLORS["refinement"]
    col_com = ARM_COLORS["commonality"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(
        f"{model}  —  {TECH_LABELS[tech]} [top-5 deflated]:  avg span-{span} gap  "
        f"⟨S_{{x+{span}}} − S_x⟩  (n={len(pairs)} pairs)\n"
        f"Pairs: {pair_str}    |    exact 95% CI",
        fontsize=9,
    )
    ax.plot(nd, ref_gap, color=col_ref, linestyle="-",  linewidth=2.2,
            label="Refinement")
    ax.fill_between(nd, ref_gap - ref_ci, ref_gap + ref_ci,
                    color=col_ref, alpha=0.20)
    ax.plot(nd, com_gap, color=col_com, linestyle="--", linewidth=2.2,
            label="Control")
    ax.fill_between(nd, com_gap - com_ci, com_gap + com_ci,
                    color=col_com, alpha=0.20)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xlabel("Normalised depth  ℓ / (L−1)")
    ax.set_ylabel(f"Δ {TECH_UNITS[tech]}  (avg over {len(pairs)} pairs)")
    ax.legend(handles=[
        plt.Line2D([0],[0], color=col_ref, linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0],[0], color=col_com, linestyle="--", linewidth=2,
                   label="Control"),
        matplotlib.patches.Patch(facecolor="dimgray", alpha=0.25,
                                 label="exact 95% CI"),
    ], fontsize=9, loc="best")
    _save(fig, f"ZY_{tech}_span{span}", model, pooling)


# ── Per-model runner ──────────────────────────────────────────────────────────

def run(model: str, pooling: str = "mean") -> None:
    print(f"\n{'='*60}\nZY — {model}  (pooling={pooling})\n{'='*60}", flush=True)

    for tech in TECHNIQUES:
        print(f"  [{model}] {TECH_LABELS[tech]}", flush=True)

        ref = _get_data(model, tech, "refinement", pooling)
        com = _get_data(model, tech, "commonality", pooling)

        if ref is None or com is None:
            print(f"  [{model}] Skipping {tech} — data unavailable.", flush=True)
            continue

        ref_chain, ref_mean, ref_sd = ref
        com_chain, com_mean, com_sd = com
        L  = ref_mean.shape[0]
        nd = normalised_depth_array(L)

        # Z4-style figures
        _fig_profiles(nd, ref_chain, ref_mean, ref_sd,
                      com_chain, com_mean, com_sd,
                      [0,1,2], tech, model, "S0_S2", pooling)
        _fig_profiles(nd, ref_chain, ref_mean, ref_sd,
                      com_chain, com_mean, com_sd,
                      [3,4],   tech, model, "S3_S4", pooling)
        _fig_gap_from_anchor(nd, ref_chain, ref_mean, ref_sd,
                             com_chain, com_mean, com_sd, 0, tech, model, pooling)
        _fig_gap_from_anchor(nd, ref_chain, ref_mean, ref_sd,
                             com_chain, com_mean, com_sd, 1, tech, model, pooling)
        _fig_gap_late_S2(nd, ref_chain, ref_mean, ref_sd,
                         com_chain, com_mean, com_sd, tech, model, pooling)
        _fig_gap_S3_S4(  nd, ref_chain, ref_mean, ref_sd,
                         com_chain, com_mean, com_sd, tech, model, pooling)

        # Z5-style figures
        for span in range(1, 5):
            _fig_span(nd, ref_chain, com_chain, span, tech, model, pooling)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZY: Z4+Z5 figures with top-5 PC deflation applied")
    parser.add_argument("--models",  nargs="+", default=MODELS)
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    args = parser.parse_args()

    for model in args.models:
        run(model, args.pooling)

    print("\nZY_top5_deflated complete.", flush=True)
