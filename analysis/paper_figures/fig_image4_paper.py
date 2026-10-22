"""
fig_image4_paper.py — Paper figure Image_4: 2 × 1 layout.

Source data: C1 constraints dataset analysis for Gemma 4 E4B (mean pooling).
Cache: results/cache/C1/gemma4-e4b_mean.npz

Top panel    : Euclidean distance, S0 as reference → distance to S1, S2, S3, S4
Bottom panel : Mahalanobis d_M,    S0 as reference → distance to S1, S2, S3, S4

Both panels contrast Refinement (solid) vs Control (dashed) with 95% CI bands.
Colours: viridis by target level (same palette as the original C1_gap figure).

Outputs
-------
  results/Images/Image_4.pdf
  results/Images/Image_4.png

Usage
-----
  python analysis/paper_figures/fig_image4_paper.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

_ROOT  = Path(__file__).resolve().parent.parent.parent
_CACHE = _ROOT / "results" / "caches" / "refinement" / "mean"
_OUT   = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import setup_plt

setup_plt()

MODEL   = "gemma4-e4b"
POOLING = "mean"

# Viridis palette at 5 stops — same as C1 analysis script
_CMAP   = matplotlib.colormaps["viridis"]
_STOPS  = [0.1, 0.3, 0.5, 0.7, 0.9]
LEVEL_COLORS = [mcolors.to_hex(_CMAP(t)) for t in _STOPS]   # S0..S4

# S0 is the reference — we plot distances to S1, S2, S3, S4 (indices 1..4)
TARGETS      = [1, 2, 3, 4]
TARGET_LABELS = {k: f"→S{k}" for k in TARGETS}

REF_LV = 0   # S0 reference row

# Arm styles
LINE_REF = "-"   # refinement — solid
LINE_COM = "--"  # control    — dashed
ALPHA_CI = 0.15
LW       = 1.8


def _load() -> dict:
    p = _CACHE / f"{MODEL}_{POOLING}.npz"
    if not p.exists():
        raise FileNotFoundError(f"C1 cache not found: {p}")
    return dict(np.load(str(p), allow_pickle=True))


def build_figure() -> plt.Figure:
    d   = _load()
    nd  = d["nd"]
    L   = int(d["L"])

    # Arrays shape: (5, 5, L) — [ref_level, tgt_level, layer]
    gap_ref_euc    = d["gap_ref_euc"]
    gap_com_euc    = d["gap_com_euc"]
    gap_ref_ci_euc = d["gap_ref_ci_euc"]
    gap_com_ci_euc = d["gap_com_ci_euc"]
    gap_ref_mah    = d["gap_ref_mah"]
    gap_com_mah    = d["gap_com_mah"]
    gap_ref_ci_mah = d["gap_ref_ci_mah"]
    gap_com_ci_mah = d["gap_com_ci_mah"]

    fig, axes = plt.subplots(1, 2, figsize=(16 + 2.0, 5), sharex=True)

    panels = [
        (axes[0], gap_ref_euc, gap_com_euc,
         gap_ref_ci_euc, gap_com_ci_euc,
         "Euclidean distance"),
        (axes[1], gap_ref_mah, gap_com_mah,
         gap_ref_ci_mah, gap_com_ci_mah,
         "Mahalanobis distance"),
    ]

    for ax, g_ref, g_com, ci_ref, ci_com, ylabel in panels:
        for tgt in TARGETS:
            col = LEVEL_COLORS[tgt]

            # Refinement — solid
            mean_r = g_ref[REF_LV, tgt]
            ci_r   = ci_ref[REF_LV, tgt]
            ax.plot(nd, mean_r, color=col, linestyle=LINE_REF, linewidth=LW,
                    label=TARGET_LABELS[tgt])
            ax.fill_between(nd, mean_r - ci_r, mean_r + ci_r,
                            color=col, alpha=ALPHA_CI)

            # Control — dashed
            mean_c = g_com[REF_LV, tgt]
            ci_c   = ci_com[REF_LV, tgt]
            ax.plot(nd, mean_c, color=col, linestyle=LINE_COM, linewidth=LW)
            ax.fill_between(nd, mean_c - ci_c, mean_c + ci_c,
                            color=col, alpha=ALPHA_CI * 0.6)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Normalised depth  ℓ / (L−1)", fontsize=10)

    # ── Legend on the right ──────────────────────────────────────────────────
    level_handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[k], linewidth=2,
                   label=f"S0 → S{k}")
        for k in TARGETS
    ]

    # Separator line
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
               bbox_to_anchor=(0.85, 0.5), bbox_transform=fig.transFigure,
               frameon=True, borderaxespad=0.2)

    fig.tight_layout(rect=(0, 0, 0.84, 1))
    return fig


if __name__ == "__main__":
    fig = build_figure()
    for ext in ("pdf", "png"):
        out = _OUT / f"Image_4.{ext}"
        fig.savefig(str(out), bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"saved → {out.relative_to(_ROOT)}")
    plt.close(fig)
    print("done.")
