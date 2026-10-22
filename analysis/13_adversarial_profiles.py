"""
13_adversarial_profiles.py — Gap profiles: Refinement vs Control vs each adversarial arm.

Produces two figures (one per adversarial arm):
  - Rows: LV (top) and PR (bottom)
  - Columns: one per model (4)
  - Three lines per panel: Refinement, Control, Adversarial arm, each with 95% CI band.

Figures saved to:
  results/figures_mean_pooling/adversarial_profiles/adv_near_synonym_profiles.pdf/.png
  results/figures_mean_pooling/adversarial_profiles/adv_scope_hedge_profiles.pdf/.png

Data sources
------------
  PR  : results/caches/refinement/mean/B1_pr_cache/{model}_{arm}_raw.npz
  LV  : results/caches/refinement/mean/B2/{model}.npz  (ref + com)
        results/caches/refinement/mean/ZY/{model}_{arm}_mean_id_lv.npz  (adv)

Usage
-----
  python analysis/13_adversarial_profiles.py
  python analysis/13_adversarial_profiles.py --pooling last
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, SEED,
    get_cache_dir, get_figures_dir,
    normalised_depth_array,
    ARM_REFINEMENT, ARM_COMMONALITY, ARM_NEAR_SYNONYM, ARM_SCOPE_HEDGE,
    ARM_COLORS, ARM_LABELS,
    save_figure, setup_plt,
)

np.random.seed(SEED)
setup_plt()

Z95     = 1.96
LW      = 1.6
ALPHA   = 0.18   # CI band opacity

MODEL_LABELS = {
    "pythia-2.8b": "Pythia 2.8B",
    "llama3.2-3b": "Llama 3.2 3B",
    "qwen2.5-3b":  "Qwen 2.5 3B",
    "gemma4-e4b":  "Gemma 4 E4B",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load(p: Path) -> dict:
    return dict(np.load(str(p), allow_pickle=True))


def _pr_gap_profile(chain_pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """chain_pr (n, L, 5) → mean gap (L,), half-CI (L,) for S4−S0."""
    per_chain = chain_pr[:, :, 4] - chain_pr[:, :, 0]   # (n, L)
    n = per_chain.shape[0]
    mean = per_chain.mean(axis=0)
    ci   = Z95 * per_chain.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, ci


def _lv_gap_profile(chain_lv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """chain_lv (n, L, 5) → mean gap (L,), half-CI (L,) for S4−S0."""
    per_chain = chain_lv[:, :, 4] - chain_lv[:, :, 0]
    n = per_chain.shape[0]
    mean = per_chain.mean(axis=0)
    ci   = Z95 * per_chain.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, ci


def load_model_data(model: str, adv_arm: str, pooling: str) -> dict:
    """Return PR and LV gap profiles for ref, com, and one adv arm."""
    b1  = get_cache_dir("refinement", pooling) / "B1_pr_cache"
    b2  = get_cache_dir("refinement", pooling) / "B2"
    zy  = get_cache_dir("refinement", pooling) / "ZY"
    tag = "last" if pooling == "last" else "mean"

    # PR
    pr_ref = _load(b1 / f"{model}_{ARM_REFINEMENT}_raw.npz")["chain_pr"]
    pr_com = _load(b1 / f"{model}_{ARM_COMMONALITY}_raw.npz")["chain_pr"]
    pr_adv = _load(b1 / f"{model}_{adv_arm}_raw.npz")["chain_pr"]

    # LV
    b2d    = _load(b2 / f"{model}.npz")
    lv_ref = b2d["ref_chain_lv"]
    lv_com = b2d["com_chain_lv"]
    lv_adv = _load(zy / f"{model}_{adv_arm}_{tag}_id_lv.npz")["chain_lv"]

    L  = pr_ref.shape[1]
    nd = normalised_depth_array(L)

    return {
        "nd": nd,
        "pr": {
            ARM_REFINEMENT: _pr_gap_profile(pr_ref),
            ARM_COMMONALITY: _pr_gap_profile(pr_com),
            adv_arm: _pr_gap_profile(pr_adv),
        },
        "lv": {
            ARM_REFINEMENT: _lv_gap_profile(lv_ref),
            ARM_COMMONALITY: _lv_gap_profile(lv_com),
            adv_arm: _lv_gap_profile(lv_adv),
        },
        "n_adv": pr_adv.shape[0],
        "n_main": pr_ref.shape[0],
    }


# ── Panel drawing ─────────────────────────────────────────────────────────────

def _draw_panel(ax: plt.Axes, nd: np.ndarray,
                profiles: dict, arms: list[str],
                ylabel: str, title: str,
                show_xlabel: bool = False) -> None:
    for arm in arms:
        mean, ci = profiles[arm]
        col = ARM_COLORS.get(arm, "gray")
        ls  = "-" if arm == ARM_REFINEMENT else ("--" if arm == ARM_COMMONALITY else "-.")
        ax.plot(nd, mean, color=col, linestyle=ls, linewidth=LW,
                label=ARM_LABELS.get(arm, arm))
        ax.fill_between(nd, mean - ci, mean + ci, color=col, alpha=ALPHA)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", zorder=0)
    if show_xlabel:
        ax.set_xlabel("Normalised depth", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


# ── Figure builder ────────────────────────────────────────────────────────────

def build_figure(adv_arm: str, pooling: str) -> plt.Figure:
    """2 rows (LV top, PR bottom) × 4 columns (models)."""
    fig, axes = plt.subplots(
        2, len(MODELS),
        figsize=(4.5 * len(MODELS), 8),
        squeeze=False,
        layout="constrained",
    )

    adv_label = ARM_LABELS.get(adv_arm, adv_arm)
    

    arms = [ARM_REFINEMENT, ARM_COMMONALITY, adv_arm]

    for col, model in enumerate(MODELS):
        data = load_model_data(model, adv_arm, pooling)
        nd   = data["nd"]
        n_main = data["n_main"]
        n_adv  = data["n_adv"]
        model_label = MODEL_LABELS[model]

        # LV — top row (no x-axis label)
        _draw_panel(
            axes[0, col], nd, data["lv"], arms,
            ylabel="Δ log-vol (S4 − S0)" if col == 0 else "",
            title=f"{model_label}",
            show_xlabel=False,
        )

        # PR — bottom row (x-axis label only here)
        _draw_panel(
            axes[1, col], nd, data["pr"], arms,
            ylabel="Δ PR (S4 − S0)" if col == 0 else "",
            title="",
            show_xlabel=True,
        )

    # Shared legend on rightmost column
    handles = [
        plt.Line2D([0], [0], color=ARM_COLORS[a],
                   linestyle=("-" if a == ARM_REFINEMENT
                               else ("--" if a == ARM_COMMONALITY else "-.")),
                   linewidth=LW, label=ARM_LABELS.get(a, a))
        for a in arms
    ]
    handles.append(mpatches.Patch(facecolor="gray", alpha=ALPHA, label="95% CI"))
    fig.legend(handles=handles, fontsize=8, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.02))

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="13: Adversarial gap profiles (LV and PR) with CI bands")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    args = parser.parse_args()

    for adv_arm in [ARM_NEAR_SYNONYM, ARM_SCOPE_HEDGE]:
        arm_slug = adv_arm.replace("adv_", "")
        print(f"Building figure for {adv_arm}...", flush=True)
        fig = build_figure(adv_arm, args.pooling)
        save_figure(fig, args.pooling, "adversarial_profiles",
                    f"{adv_arm}_profiles")
        plt.close(fig)
        print(f"  saved {adv_arm}", flush=True)

    print("\n13 complete.", flush=True)
