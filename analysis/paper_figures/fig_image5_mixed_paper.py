"""
fig_image5_mixed_paper.py — Paper figure Image_5_mixed: 1 × 2 layout.

Left panel  : Δ Log-volume gap from S0  — top-5 PC deflated  (mean pooling)
              (same as left panel of Image_5_top5)
Right panel : Δ Participation Ratio gap from S0 — raw, no deflation  (last-token)
              (same as right panel of Image_5_raw)

Model: LLaMA 3.2 3B

Data sources
------------
  Log-vol top5 : results/cache/ZY/llama3.2-3b_{arm}_mean_id_lv.npz
  PR raw       : results/cache_last/B1_pr_cache/llama3.2-3b_{arm}_raw.npz

Output
------
  results/Images/Image_5_mixed.pdf
  results/Images/Image_5_mixed.png

Usage
-----
  python analysis/paper_figures/fig_image5_mixed_paper.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

_ROOT     = Path(__file__).resolve().parent.parent.parent
_ZY_CACHE = _ROOT / "results" / "caches" / "refinement" / "mean" / "ZY"
_B1_LAST  = _ROOT / "results" / "caches" / "refinement" / "last" / "B1_pr_cache"
_OUT      = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import normalised_depth_array, setup_plt

setup_plt()

MODEL    = "llama3.2-3b"
N_CHAINS = 20
Z_95     = 1.96

_CMAP        = matplotlib.colormaps["viridis"]
LEVEL_COLORS = [mcolors.to_hex(_CMAP(t)) for t in [0.1, 0.3, 0.5, 0.7, 0.9]]

TARGETS = [1, 2, 3, 4]
LW      = 1.8
ALPHA   = 0.15


# ── Data loaders ──────────────────────────────────────────────────────────────

def _lv_top5(arm: str) -> tuple[np.ndarray, int]:
    p = _ZY_CACHE / f"{MODEL}_{arm}_mean_id_lv.npz"
    d = dict(np.load(str(p), allow_pickle=True))
    chain_lv = d["chain_lv"]   # (20, L, 5)
    return chain_lv, chain_lv.shape[1]


def _pr_raw(arm: str) -> tuple[np.ndarray, np.ndarray, int]:
    p = _B1_LAST / f"{MODEL}_{arm}_raw.npz"
    d = dict(np.load(str(p), allow_pickle=True))
    return d["mean_pr"], d["sd_pr"], int(d["mean_pr"].shape[0])


# ── Gap computation ───────────────────────────────────────────────────────────

def _lv_gap(chain_lv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact CI: per-chain differences. Returns (gap_mean, gap_ci) each (L, 4)."""
    L = chain_lv.shape[1]
    gap_mean = np.zeros((L, len(TARGETS)))
    gap_ci   = np.zeros((L, len(TARGETS)))
    for i, k in enumerate(TARGETS):
        per_chain      = chain_lv[:, :, k] - chain_lv[:, :, 0]
        gap_mean[:, i] = per_chain.mean(axis=0)
        gap_ci[:, i]   = Z_95 * per_chain.std(axis=0, ddof=1) / np.sqrt(N_CHAINS)
    return gap_mean, gap_ci


def _pr_gap(mean_pr: np.ndarray, sd_pr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conservative CI. Returns (gap_mean, gap_ci) each (L, 4)."""
    L = mean_pr.shape[0]
    gap_mean = np.zeros((L, len(TARGETS)))
    gap_ci   = np.zeros((L, len(TARGETS)))
    for i, k in enumerate(TARGETS):
        gap_mean[:, i] = mean_pr[:, k] - mean_pr[:, 0]
        gap_ci[:, i]   = Z_95 * np.sqrt(
            (sd_pr[:, k] / np.sqrt(N_CHAINS)) ** 2 +
            (sd_pr[:, 0] / np.sqrt(N_CHAINS)) ** 2
        )
    return gap_mean, gap_ci


# ── Panel drawing ─────────────────────────────────────────────────────────────

def _draw_panel(ax, nd, gap_ref, ci_ref, gap_com, ci_com, ylabel):
    for i, k in enumerate(TARGETS):
        col = LEVEL_COLORS[k]
        ax.plot(nd, gap_ref[:, i], color=col, linestyle="-",  linewidth=LW)
        ax.fill_between(nd,
                        gap_ref[:, i] - ci_ref[:, i],
                        gap_ref[:, i] + ci_ref[:, i],
                        color=col, alpha=ALPHA)
        ax.plot(nd, gap_com[:, i], color=col, linestyle="--", linewidth=LW)
        ax.fill_between(nd,
                        gap_com[:, i] - ci_com[:, i],
                        gap_com[:, i] + ci_com[:, i],
                        color=col, alpha=ALPHA * 0.6)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("Normalised depth  ℓ / (L−1)", fontsize=10)


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16 + 2.0, 5))

    # ── Left: log-volume top-5 deflated ──────────────────────────────────────
    chain_ref, L = _lv_top5("refinement")
    chain_com, _ = _lv_top5("commonality")
    nd_lv = normalised_depth_array(L)
    lv_gap_ref, lv_ci_ref = _lv_gap(chain_ref)
    lv_gap_com, lv_ci_com = _lv_gap(chain_com)
    _draw_panel(axes[0], nd_lv, lv_gap_ref, lv_ci_ref, lv_gap_com, lv_ci_com,
                "Δ Log-volume")

    # ── Right: participation ratio raw ────────────────────────────────────────
    mean_ref, sd_ref, L_pr = _pr_raw("refinement")
    mean_com, sd_com, _    = _pr_raw("commonality")
    nd_pr = normalised_depth_array(L_pr)
    pr_gap_ref, pr_ci_ref = _pr_gap(mean_ref, sd_ref)
    pr_gap_com, pr_ci_com = _pr_gap(mean_com, sd_com)
    _draw_panel(axes[1], nd_pr, pr_gap_ref, pr_ci_ref, pr_gap_com, pr_ci_com,
                "Δ Participation Ratio")

    # ── Legend on the right ──────────────────────────────────────────────────
    level_handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[k], linewidth=2,
                   label=f"S0 → S{k}")
        for k in TARGETS
    ]
    separator = plt.Line2D([0], [0], color="none", label="")
    style_handles = [
        plt.Line2D([0], [0], color="dimgray", linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0], [0], color="dimgray", linestyle="--", linewidth=2,
                   label="Control"),
        mpatches.Patch(facecolor="dimgray", alpha=0.20, label="95% CI"),
    ]
    fig.legend(handles=level_handles + [separator] + style_handles,
               fontsize=8.5, ncol=1, loc="center left",
               bbox_to_anchor=(0.84, 0.5), bbox_transform=fig.transFigure,
               frameon=True, borderaxespad=0.2)

    fig.tight_layout(rect=[0, 0, 0.84, 1])
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig = build_figure()
    for ext in ("pdf", "png"):
        out = _OUT / f"Image_5_mixed.{ext}"
        fig.savefig(str(out), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"saved → {out.relative_to(_ROOT)}")
    plt.close(fig)
    print("done.")
