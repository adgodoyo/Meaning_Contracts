"""
04_participation_ratio.py — Participation Ratio Profiles
Reproduces Figure 4. Two-stage: gap profile → peak layer → level breakdown.

Branches
--------
  raw       : raw activations
  top5      : global top-5 PC deflation per layer (PCs from arm activations)
  lang_resid: surface-template residualization per layer (16 templates ≡ languages)

Outputs
-------
results/figures/per_model/{model}_B1_participation_ratio.pdf
results/figures/aggregate/AGG_B1_pr_gap_all_models.pdf
results/tables/B1_peak_layers.csv
results/tables/B1_pr_per_transition.csv   ← fixes "0.5 PR units" claim
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, MODEL_COLORS, TABLES_DIR,
    ARM_REFINEMENT, ARM_COMMONALITY, ARMS_ALL, ARM_LABELS, ARM_COLORS,
    load_activations, load_constraints_labels,
    participation_ratio,
    normalised_depth_array, normalised_depth,
    get_cache_dir,
    gap_profile, peak_layer,
    model_linestyle, save_figure, setup_plt, SEED,
)


def _get_b1_cache(pooling: str) -> Path:
    d = get_cache_dir("refinement", pooling) / "B1_pr_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(model: str, arm: str, branch: str, pooling: str = "mean") -> Path:
    return _get_b1_cache(pooling) / f"{model}_{arm}_{branch}.npz"


def _save_pr(model: str, arm: str, branch: str,
             mean_pr: np.ndarray, sd_pr: np.ndarray,
             chain_pr: np.ndarray | None = None,
             pooling: str = "mean") -> None:
    kwargs = dict(mean_pr=mean_pr, sd_pr=sd_pr)
    if chain_pr is not None:
        kwargs["chain_pr"] = chain_pr
    np.savez(_cache_path(model, arm, branch, pooling), **kwargs)


def _load_pr(model: str, arm: str,
             branch: str, pooling: str = "mean") -> tuple[np.ndarray, np.ndarray] | None:
    p = _cache_path(model, arm, branch, pooling)
    if not p.exists():
        return None
    d = np.load(str(p))
    # Recompute if per-chain array is missing (needed for exact CI in Z5)
    if "chain_pr" not in d:
        return None
    return d["mean_pr"], d["sd_pr"]


def _load_chain_pr(model: str, arm: str, branch: str, pooling: str = "mean") -> np.ndarray | None:
    """Return per-chain PR array (n_chains, L, 5), or None if not cached."""
    p = _cache_path(model, arm, branch, pooling)
    if not p.exists():
        return None
    d = np.load(str(p))
    return d["chain_pr"] if "chain_pr" in d else None

np.random.seed(SEED)
setup_plt()

LEVELS  = [0, 1, 2, 3, 4]
BRANCHES = ["raw", "top5", "lang_resid"]
BRANCH_LABELS = {
    "raw":       "Raw",
    "top5":      "Top-5 deflated",
    "lang_resid":"Template residualized",
}
BRANCH_LS = {
    "raw":       "-",
    "top5":      "--",
    "lang_resid": ":",
}
K_DEFLATE = 5


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute per-arm transform parameters (once per arm, used by all branches)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_arm_transforms(acts: np.ndarray, mask_arm: np.ndarray,
                               clab: dict) -> dict:
    """
    Compute per-layer:
      arm_mean      (L, D)  — mean of all arm sentences
      Vt_top5       (L, k, D)  — top-k right singular vectors (arm-centred)
      template_means (L, T, D) — per surface_variant mean
      template_list  (T,)   — sorted unique template ids
    Processes one layer at a time to avoid loading the full arm into memory.
    """
    L, D = acts.shape[1], acts.shape[2]
    arm_idx = np.where(mask_arm)[0]
    N_arm   = len(arm_idx)

    template_list = np.sort(np.unique(clab["surface_variant"][mask_arm]))
    T             = len(template_list)
    tmpl_labels   = clab["surface_variant"][mask_arm]   # (N_arm,)

    arm_mean       = np.zeros((L, D),        dtype=np.float64)
    Vt_top5        = np.zeros((L, K_DEFLATE, D), dtype=np.float64)
    template_means = np.zeros((L, T, D),     dtype=np.float64)

    print(f"    precomputing branch transforms ({L} layers)...", flush=True)
    for l in range(L):
        if l % 5 == 0:
            print(f"      layer {l}/{L}", flush=True)
        X_l = acts[arm_idx, l, :].astype(np.float64)   # (N_arm, D)
        arm_mean[l] = X_l.mean(axis=0)
        Xc_l = X_l - arm_mean[l]
        k_eff = min(K_DEFLATE, N_arm - 1, D)
        _, _, Vt = randomized_svd(Xc_l, n_components=k_eff, random_state=SEED)
        Vt_top5[l, :k_eff] = Vt
        for ti, t in enumerate(template_list):
            mask_t = tmpl_labels == t
            template_means[l, ti] = X_l[mask_t].mean(axis=0)

    return {
        "arm_mean":       arm_mean,
        "Vt_top5":        Vt_top5,
        "template_means": template_means,
        "template_list":  template_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Apply branch transformation to a group's (L, n, D) tensor
# ─────────────────────────────────────────────────────────────────────────────

def apply_branch(X_ch: np.ndarray, mask_ch: np.ndarray,
                 clab: dict, branch: str, transforms: dict) -> np.ndarray:
    """
    X_ch:    (L, n, D) float64 — raw group activations
    Returns: (L, n, D) float64 — transformed group activations
    """
    if branch == "raw":
        return X_ch

    L, n, D = X_ch.shape

    if branch == "top5":
        arm_mean = transforms["arm_mean"]   # (L, D)
        Vt       = transforms["Vt_top5"]    # (L, k, D)
        # Centre relative to arm mean, project out top-5, leave residual
        Xc_arm = X_ch - arm_mean[:, None, :]           # (L, n, D)
        proj   = np.einsum("lnd,lkd->lnk", Xc_arm, Vt) # (L, n, k)
        recon  = np.einsum("lnk,lkd->lnd", proj, Vt)   # (L, n, D)
        return Xc_arm - recon                            # (L, n, D) deflated

    if branch == "lang_resid":
        tmpl_list   = transforms["template_list"]
        tmpl_means  = transforms["template_means"]  # (L, T, D)
        tmpl_ids    = clab["surface_variant"][mask_ch]  # (n,)
        # Map each sentence's template id to its position in tmpl_list
        tmpl_pos = np.searchsorted(tmpl_list, tmpl_ids)  # (n,)
        tmpl_means_group = tmpl_means[:, tmpl_pos, :]    # (L, n, D)
        return X_ch - tmpl_means_group                   # (L, n, D) residualized

    raise ValueError(f"Unknown branch: {branch}")


# ─────────────────────────────────────────────────────────────────────────────
# PR matrix per arm: (L, 5 levels) mean ± SD across chains, all branches
# ─────────────────────────────────────────────────────────────────────────────

def pr_by_level_layer_all_branches(
    acts: np.ndarray, clab: dict, arm: str, model: str = "",
    branches_to_compute: list[str] | None = None,
    pooling: str = "mean",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns {branch: (mean_pr (L,5), sd_pr (L,5))} for all BRANCHES.
    Loads from cache where available; only computes missing branches.
    """
    if branches_to_compute is None:
        branches_to_compute = BRANCHES

    # ── Load cached branches ──────────────────────────────────────────────
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    need_compute: list[str] = []
    for br in branches_to_compute:
        cached = _load_pr(model, arm, br, pooling)
        if cached is not None:
            print(f"    [{arm}/{br}] loaded from cache", flush=True)
            results[br] = cached
        else:
            need_compute.append(br)

    if not need_compute:
        return results

    L = acts.shape[1]

    # ── Pre-compute branch transforms once for this arm ───────────────────
    mask_arm = clab["arm"] == arm
    chains   = np.unique(clab["chain_id"][mask_arm])
    n_chains = len(chains)

    transforms = precompute_arm_transforms(acts, mask_arm, clab)

    # Storage per branch that needs computing
    branch_pr: dict[str, np.ndarray] = {
        br: np.zeros((n_chains, L, 5)) for br in need_compute
    }

    for li, level in enumerate(LEVELS):
        mask_lv = mask_arm & (clab["level"] == level)
        print(f"    [{arm}] level {level} ({li+1}/5, {n_chains} chains)...",
              flush=True)

        for ci, ch in enumerate(chains):
            if ci % 5 == 0:
                print(f"      chain {ci+1}/{n_chains}", flush=True)
            mask_ch = mask_lv & (clab["chain_id"] == ch)
            n_sent  = mask_ch.sum()
            if n_sent < 2:
                for br in need_compute:
                    branch_pr[br][ci, :, li] = np.nan
                continue

            # Load raw group activations once — (L, n, D)
            X_ch_raw = (acts[mask_ch]
                        .astype(np.float64)
                        .transpose(1, 0, 2))          # (L, n, D)

            for br in need_compute:
                X_br = apply_branch(X_ch_raw, mask_ch, clab, br, transforms)
                Xc   = X_br - X_br.mean(axis=1, keepdims=True)  # (L, n, D)
                S    = np.linalg.svd(Xc, compute_uv=False,
                                     full_matrices=False)         # (L, min(n,D))
                S2       = S ** 2
                sum_s2   = S2.sum(axis=1)         # (L,)
                sum_s4   = (S2 ** 2).sum(axis=1)  # (L,)
                valid    = sum_s4 > 0
                pr_vals  = np.where(valid, sum_s2 ** 2 / sum_s4, 0.0)
                branch_pr[br][ci, :, li] = pr_vals

    for br in need_compute:
        data = branch_pr[br]   # (n_chains, L, 5)
        mean_pr = np.nanmean(data, axis=0)  # (L, 5)
        sd_pr   = np.nanstd(data,  axis=0, ddof=1)
        _save_pr(model, arm, br, mean_pr, sd_pr, chain_pr=data, pooling=pooling)
        results[br] = (mean_pr, sd_pr)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-model computation
# ─────────────────────────────────────────────────────────────────────────────

def run(model: str, pooling: str = "mean",
        branches_to_compute: list[str] | None = None) -> dict:
    print(f"\n{'='*60}\nB1 — {model}\n{'='*60}")

    if branches_to_compute is None:
        branches_to_compute = BRANCHES

    clab  = load_constraints_labels()
    available_arms_check = [a for a in ARMS_ALL if (clab["arm"] == a).sum() > 0]

    # Check if all required caches exist — if so, skip loading activations
    all_cached = all(
        _cache_path(model, arm, br, pooling).exists()
        for arm in available_arms_check
        for br in branches_to_compute
    )

    if all_cached:
        # Get L from one of the cache files
        sample_path = _cache_path(model, available_arms_check[0], branches_to_compute[0], pooling)
        _tmp = dict(np.load(str(sample_path), allow_pickle=True))
        L = int(_tmp["mean_pr"].shape[0])
        acts = None
        print(f"  [{model}] All B1 caches present — skipping activation load (L={L})", flush=True)
    else:
        acts  = load_activations(model, "constraints", pooling)
        _, L, _ = acts.shape

    nd    = normalised_depth_array(L)

    available_arms = available_arms_check
    print(f"  Available arms: {available_arms}")

    # pr_data[arm][branch] = (mean_pr (L,5), sd_pr (L,5))
    pr_data: dict = {}
    # gaps[arm][branch] = (L,) gap profile
    gaps: dict    = {}
    peaks: dict   = {}

    for arm in available_arms:
        print(f"  Computing PR for arm '{arm}'...", flush=True)
        pr_data[arm] = pr_by_level_layer_all_branches(
            acts, clab, arm, model=model,
            branches_to_compute=branches_to_compute,
            pooling=pooling,
        )
        # Gap profile and peak from raw branch
        mean_pr_raw = pr_data[arm]["raw"][0]
        g = gap_profile(mean_pr_raw, high_idx=4, low_idx=0)
        gaps[arm]  = g
        peaks[arm] = peak_layer(g, smooth=3)
        print(f"    Peak layer (raw): {peaks[arm]}  "
              f"(ND {normalised_depth(peaks[arm],L):.2f})  "
              f"gap={g[peaks[arm]]:.3f}")

    lstar = peaks.get(ARM_REFINEMENT, peaks[available_arms[0]])

    # ── Per-transition PR gain at lstar (raw, refinement arm) ────────────────
    transitions = {}
    if ARM_REFINEMENT in pr_data:
        mean_ref = pr_data[ARM_REFINEMENT]["raw"][0]
        for i in range(4):
            delta = mean_ref[lstar, i + 1] - mean_ref[lstar, i]
            transitions[f"S{i}→S{i+1}"] = delta
        print(f"  Per-transition PR gain at layer {lstar}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in transitions.items()))

    # ── Per-model figure (4 panels) ───────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(8, 14))
    fig.suptitle(f"{model}  —  Participation Ratio (raw / top-5 / template-resid)",
                 fontsize=10)

    # Panel 0: raw gap profiles for all arms
    ax = axes[0]
    for arm in available_arms:
        g  = gaps[arm]
        ls = "-" if arm == ARM_REFINEMENT else ("--" if arm == ARM_COMMONALITY else ":")
        ax.plot(nd, g, color=ARM_COLORS.get(arm, "gray"),
                linestyle=ls, label=ARM_LABELS.get(arm, arm), linewidth=1.5)
    ax.axvline(normalised_depth(lstar, L), color="red", linestyle=":", linewidth=1.0,
               label=f"layer* = {lstar} (ND={normalised_depth(lstar,L):.2f})")
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("PR(S4) − PR(S0)")
    ax.set_title("Stage 1: PR gap profile (raw branch)")
    ax.legend(fontsize=7, ncol=2)

    # Panel 1: refinement — available branches at lstar
    ax1 = axes[1]
    if ARM_REFINEMENT in pr_data:
        for br in BRANCHES:
            if br not in pr_data[ARM_REFINEMENT]:
                continue
            mean_pr, sd_pr = pr_data[ARM_REFINEMENT][br]
            ax1.errorbar(LEVELS, mean_pr[lstar], yerr=sd_pr[lstar],
                         color=ARM_COLORS[ARM_REFINEMENT],
                         linestyle=BRANCH_LS[br], marker="o", capsize=3,
                         label=BRANCH_LABELS[br], linewidth=1.5)
    ax1.set_xlabel("Constraint level S0–S4")
    ax1.set_ylabel("Participation Ratio")
    ax1.set_title(
        f"{ARM_LABELS[ARM_REFINEMENT]}  — layer*={lstar} "
        f"(ND={normalised_depth(lstar,L):.2f})"
    )
    ax1.set_xticks(LEVELS)
    ax1.legend(fontsize=7)

    # Panel 2: commonality — available branches at lstar
    ax2 = axes[2]
    if ARM_COMMONALITY in pr_data:
        for br in BRANCHES:
            if br not in pr_data[ARM_COMMONALITY]:
                continue
            mean_pr, sd_pr = pr_data[ARM_COMMONALITY][br]
            ax2.errorbar(LEVELS, mean_pr[lstar], yerr=sd_pr[lstar],
                         color=ARM_COLORS[ARM_COMMONALITY],
                         linestyle=BRANCH_LS[br], marker="o", capsize=3,
                         label=BRANCH_LABELS[br], linewidth=1.5)
    ax2.set_xlabel("Constraint level S0–S4")
    ax2.set_ylabel("Participation Ratio")
    ax2.set_title(
        f"{ARM_LABELS[ARM_COMMONALITY]}  — layer*={lstar} "
        f"(ND={normalised_depth(lstar,L):.2f})"
    )
    ax2.set_xticks(LEVELS)
    ax2.legend(fontsize=7)

    # Panel 3: adversarial arms — raw only
    ax3 = axes[3]
    adv_arms = [a for a in available_arms
                if a not in (ARM_REFINEMENT, ARM_COMMONALITY)]
    for arm in adv_arms:
        mean_pr, sd_pr = pr_data[arm]["raw"]
        ax3.errorbar(LEVELS, mean_pr[lstar], yerr=sd_pr[lstar],
                     color=ARM_COLORS.get(arm, "gray"), marker="s", capsize=3,
                     label=ARM_LABELS.get(arm, arm), linewidth=1.2)
    if ARM_REFINEMENT in pr_data:
        mean_ref, _ = pr_data[ARM_REFINEMENT]["raw"]
        ax3.plot(LEVELS, mean_ref[lstar], "b-o", linewidth=1.2, alpha=0.4,
                 label=f"{ARM_LABELS[ARM_REFINEMENT]} (ref, raw)")
    ax3.set_xlabel("Constraint level S0–S4")
    ax3.set_ylabel("Participation Ratio")
    ax3.set_title("Adversarial arms at layer* (raw branch)")
    ax3.set_xticks(LEVELS)
    ax3.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, pooling, "refinement_pr", f"{model}_B1_participation_ratio")

    return {
        "model": model, "L": L, "nd": nd,
        "pr_data": pr_data, "gaps": gaps, "peaks": peaks,
        "lstar": lstar, "transitions": transitions,
        "available_arms": available_arms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate figure
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(all_results: list, pooling: str = "mean") -> None:
    print("\nB1 — Aggregate gap profile figure")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PR gap profile (raw) — all models (refinement vs commonality)",
                 fontsize=11)

    for ax, arm in [(axes[0], ARM_REFINEMENT), (axes[1], ARM_COMMONALITY)]:
        for res in all_results:
            if arm not in res["gaps"]:
                continue
            ax.plot(res["nd"], res["gaps"][arm],
                    color=MODEL_COLORS[res["model"]],
                    linestyle=model_linestyle(res["model"]),
                    label=res["model"], linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("Normalised depth  ℓ/(L−1)")
        ax.set_ylabel("PR(S4) − PR(S0)")
        ax.set_title(ARM_LABELS.get(arm, arm))
        ax.legend()

    save_figure(fig, pooling, "refinement_pr", "AGG_B1_pr_gap_all_models")


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def save_tables(all_results: list) -> None:
    # Peak layers (from raw branch)
    peak_path = TABLES_DIR / "B1_peak_layers.csv"
    with open(peak_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "arm", "peak_layer", "peak_nd", "gap_at_peak"])
        for res in all_results:
            L = res["L"]
            for arm, lstar in res["peaks"].items():
                gap_val = res["gaps"][arm][lstar]
                w.writerow([res["model"], arm, lstar,
                             f"{normalised_depth(lstar,L):.4f}",
                             f"{gap_val:.4f}"])
    print(f"  Peak layers → {peak_path.name}")

    # Per-transition gains (raw branch, refinement arm)
    trans_path = TABLES_DIR / "B1_pr_per_transition.csv"
    with open(trans_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "lstar", "S0→S1", "S1→S2", "S2→S3", "S3→S4", "total"])
        for res in all_results:
            t = res["transitions"]
            if not t:
                continue
            total = sum(t.values())
            w.writerow([res["model"], res["lstar"],
                        f"{t.get('S0→S1',0):.4f}",
                        f"{t.get('S1→S2',0):.4f}",
                        f"{t.get('S2→S3',0):.4f}",
                        f"{t.get('S3→S4',0):.4f}",
                        f"{total:.4f}"])
    print(f"  Per-transition gains → {trans_path.name}")

    # PR values per branch at lstar for refinement arm
    branch_path = TABLES_DIR / "B1_pr_by_branch.csv"
    with open(branch_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "arm", "branch", "lstar",
                    "pr_S0", "pr_S4", "gap_S4_S0"])
        for res in all_results:
            lstar = res["lstar"]
            for arm in [ARM_REFINEMENT, ARM_COMMONALITY]:
                if arm not in res["pr_data"]:
                    continue
                for br in BRANCHES:
                    mean_pr, _ = res["pr_data"][arm][br]
                    w.writerow([res["model"], arm, br, lstar,
                                f"{mean_pr[lstar, 0]:.4f}",
                                f"{mean_pr[lstar, 4]:.4f}",
                                f"{mean_pr[lstar, 4] - mean_pr[lstar, 0]:.4f}"])
    print(f"  PR by branch → {branch_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_plot_only(model: str, pooling: str = "mean",
                  branches: list[str] | None = None) -> dict:
    """Load all PR data from cache and regenerate figures without recomputing."""
    if branches is None:
        branches = BRANCHES
    acts  = load_activations(model, "constraints", pooling)
    N, L, D = acts.shape
    nd    = normalised_depth_array(L)
    clab  = load_constraints_labels()

    available_arms = [a for a in ARMS_ALL if (clab["arm"] == a).sum() > 0]

    pr_data: dict = {}
    gaps: dict    = {}
    peaks: dict   = {}
    transitions: dict = {}

    for arm in available_arms:
        arm_branches = {}
        for br in branches:
            cached = _load_pr(model, arm, br, pooling)
            if cached is not None:
                arm_branches[br] = cached
            else:
                print(f"  WARNING: no cache for {model}/{arm}/{br} — skipping",
                      flush=True)
        if not arm_branches:
            continue
        pr_data[arm] = arm_branches
        # Gap profile from raw if available, else first branch
        ref_br = "raw" if "raw" in arm_branches else next(iter(arm_branches))
        mean_pr_raw = arm_branches[ref_br][0]
        g = gap_profile(mean_pr_raw, high_idx=4, low_idx=0)
        gaps[arm]  = g
        peaks[arm] = peak_layer(g, smooth=3)

    if not pr_data:
        print(f"  No cached data found for {model}. Run without --plot-only first.",
              flush=True)
        return {}

    lstar = peaks.get(ARM_REFINEMENT, peaks[next(iter(peaks))])

    if ARM_REFINEMENT in pr_data and "raw" in pr_data[ARM_REFINEMENT]:
        mean_ref = pr_data[ARM_REFINEMENT]["raw"][0]
        for i in range(4):
            delta = mean_ref[lstar, i + 1] - mean_ref[lstar, i]
            transitions[f"S{i}→S{i+1}"] = delta

    # Reuse the same figure-drawing code path
    res = {
        "model": model, "L": L, "nd": nd,
        "pr_data": pr_data, "gaps": gaps, "peaks": peaks,
        "lstar": lstar, "transitions": transitions,
        "available_arms": list(pr_data.keys()),
    }

    # Replicate the per-model figure block from run()
    fig, axes = plt.subplots(4, 1, figsize=(8, 14))
    fig.suptitle(f"{model}  —  Participation Ratio (raw / top-5 / template-resid)",
                 fontsize=10)

    ax = axes[0]
    for arm in res["available_arms"]:
        g  = gaps[arm]
        ls = "-" if arm == ARM_REFINEMENT else ("--" if arm == ARM_COMMONALITY else ":")
        ax.plot(nd, g, color=ARM_COLORS.get(arm, "gray"),
                linestyle=ls, label=ARM_LABELS.get(arm, arm), linewidth=1.5)
    ax.axvline(normalised_depth(lstar, L), color="red", linestyle=":", linewidth=1.0,
               label=f"layer* = {lstar} (ND={normalised_depth(lstar,L):.2f})")
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_ylabel("PR(S4) − PR(S0)")
    ax.set_title("Stage 1: PR gap profile (raw branch)")
    ax.legend(fontsize=7, ncol=2)

    for ax_i, arm in [(axes[1], ARM_REFINEMENT), (axes[2], ARM_COMMONALITY)]:
        if arm not in pr_data:
            ax_i.set_visible(False)
            continue
        for br in BRANCHES:
            if br not in pr_data[arm]:
                continue
            mean_pr, sd_pr = pr_data[arm][br]
            ax_i.errorbar(LEVELS, mean_pr[lstar], yerr=sd_pr[lstar],
                          color=ARM_COLORS[arm],
                          linestyle=BRANCH_LS[br], marker="o", capsize=3,
                          label=BRANCH_LABELS[br], linewidth=1.5)
        ax_i.set_xlabel("Constraint level S0–S4")
        ax_i.set_ylabel("Participation Ratio")
        ax_i.set_title(
            f"{ARM_LABELS[arm]}  — layer*={lstar} "
            f"(ND={normalised_depth(lstar,L):.2f})"
        )
        ax_i.set_xticks(LEVELS)
        ax_i.legend(fontsize=7)

    ax3 = axes[3]
    adv_arms = [a for a in res["available_arms"]
                if a not in (ARM_REFINEMENT, ARM_COMMONALITY)]
    for arm in adv_arms:
        if "raw" not in pr_data[arm]:
            continue
        mean_pr, sd_pr = pr_data[arm]["raw"]
        ax3.errorbar(LEVELS, mean_pr[lstar], yerr=sd_pr[lstar],
                     color=ARM_COLORS.get(arm, "gray"), marker="s", capsize=3,
                     label=ARM_LABELS.get(arm, arm), linewidth=1.2)
    if ARM_REFINEMENT in pr_data and "raw" in pr_data[ARM_REFINEMENT]:
        mean_ref, _ = pr_data[ARM_REFINEMENT]["raw"]
        ax3.plot(LEVELS, mean_ref[lstar], "b-o", linewidth=1.2, alpha=0.4,
                 label=f"{ARM_LABELS[ARM_REFINEMENT]} (ref, raw)")
    ax3.set_xlabel("Constraint level S0–S4")
    ax3.set_ylabel("Participation Ratio")
    ax3.set_title("Adversarial arms at layer* (raw branch)")
    ax3.set_xticks(LEVELS)
    ax3.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, pooling, "refinement_pr", f"{model}_B1_participation_ratio")
    print(f"  [{model}] figure saved", flush=True)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B1: Participation ratio profiles (raw + top5 + template-resid)")
    parser.add_argument("--models",  nargs="+", default=MODELS)
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    parser.add_argument(
        "--branches", nargs="+", default=BRANCHES, choices=BRANCHES,
        help="Which branches to compute/show (default: all three). "
             "Pass 'top5 lang_resid' to skip recomputing raw.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Load all data from cache and regenerate figures only (no computation).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute all branches even if cache exists.",
    )
    args = parser.parse_args()

    all_results = []
    for model in args.models:
        if args.plot_only:
            res = run_plot_only(model, args.pooling, args.branches)
        else:
            branches = None if args.force else args.branches
            res = run(model, args.pooling, branches_to_compute=branches)
        if res:
            all_results.append(res)

    if all_results:
        aggregate(all_results, args.pooling)
        save_tables(all_results)
    print("\nB1 complete.")
