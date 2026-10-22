"""
fig_image1_paper.py — Paper figure Image_1: Z2 cosine similarity, 2 × 3 layout.

Same content as fig_z2_cosine_paper.py with one change:
  Raw column (col 0) y-axis fixed to [0.8, 1.0].
  Top-1 and Top-5 deflated columns use auto y-axis.

Layout
------
  Rows    : llama3.2-3b  |  qwen2.5-3b
  Columns : Raw  |  Top-1 deflated  |  Top-5 deflated

Category relabelling for the paper (legend labels only):
  Code C1 (dark-red solid)  →  paper label C2  "Same factor, same language"
  Code C2 (red   dashed)    →  paper label C1  "Exact translation, diff language"
  C3 … C7 unchanged.

Output
------
  results/Images/Image_1.pdf
  results/Images/Image_1.png

Usage
-----
  python analysis/paper_figures/fig_image1_paper.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT   = Path(__file__).resolve().parent.parent.parent
_CACHE  = _ROOT / "results" / "caches" / "multilingual" / "mean"
_OUT    = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import normalised_depth_array, setup_plt

setup_plt()

MODELS = {
    "llama3.2-3b": "LLaMA 3.2 (3B)",
    "qwen2.5-3b":  "Qwen 2.5 (3B)",
}
POOLING = "mean"

# Per-row y-limits for the raw column (keyed by model key, None = auto)
RAW_YLIM = {
    "llama3.2-3b": (0.85, 1.0),
    "qwen2.5-3b":  (0.75, 1.0),
}

COLUMNS = [
    ("raw",  "Raw"),
    ("top1", "Top-1 deflated"),
    ("top5", "Top-5 deflated"),
]

CATS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

CAT_STYLE: dict[str, tuple[str, str, float]] = {
    "C1": ("#c0392b", "-",   1.8),
    "C2": ("#e74c3c", "--",  2.2),
    "C3": ("#e67e22", "--",  1.8),
    "C4": ("#27ae60", "-",   1.8),
    "C5": ("#1abc9c", "--",  1.8),
    "C6": ("#2980b9", "-",   1.8),
    "C7": ("#154360", "--",  1.8),
}

CAT_LABEL_PAPER: dict[str, str] = {
    "C1": "C2  Same factor, same language",
    "C2": "C1  Exact translation, diff language",
    "C3": "C3  Same factor, diff language",
    "C4": "C4  Same domain, diff factor, same lang",
    "C5": "C5  Same domain, diff factor, diff lang",
    "C6": "C6  Different domain, same language",
    "C7": "C7  Different domain, diff language",
}

LEGEND_ORDER = ["C2", "C1", "C3", "C4", "C5", "C6", "C7"]


def _load_cache(model: str) -> dict:
    for p in [
        _CACHE / f"{model}_{POOLING}_R128.npz",
        _CACHE / f"{model}_{POOLING}.npz",
    ]:
        if p.exists():
            return dict(np.load(str(p), allow_pickle=True))
    raise FileNotFoundError(
        f"Z2 cache not found for {model}. Run Z2_deflation_grids.py first."
    )


def build_figure() -> plt.Figure:
    n_rows = len(MODELS)
    n_cols = len(COLUMNS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols + 1.5, 3.8 * n_rows),  # extra width for right legend
        sharex=True, sharey=False,
        squeeze=False,
    )

    for row_idx, (model_key, model_label) in enumerate(MODELS.items()):
        cache = _load_cache(model_key)
        L     = int(cache["L"])
        nd    = normalised_depth_array(L)

        for col_idx, (cond_key, cond_label) in enumerate(COLUMNS):
            ax = axes[row_idx, col_idx]

            for cat in CATS:
                color, ls, lw = CAT_STYLE[cat]
                ax.plot(nd, cache[f"cos_{cond_key}_{cat}"],
                        color=color, linestyle=ls, linewidth=lw)

            # Per-row y-limit for the raw column only
            if cond_key == "raw":
                ax.set_ylim(*RAW_YLIM[model_key])

            if col_idx == 0:
                ax.set_ylabel("Cosine similarity", fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Normalised depth", fontsize=9)

            if col_idx == 0:
                ax.set_title(f"{model_label} — {cond_label}", fontsize=9)
            else:
                ax.set_title(cond_label, fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0],
                   color=CAT_STYLE[cat][0],
                   linestyle=CAT_STYLE[cat][1],
                   linewidth=CAT_STYLE[cat][2],
                   label=CAT_LABEL_PAPER[cat])
        for cat in LEGEND_ORDER
    ]

    # Vertical legend on the right, outside the plot area
    fig.legend(
        handles=legend_handles,
        fontsize=8,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        bbox_transform=fig.transFigure,
        frameon=True,
        borderaxespad=0,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    return fig


if __name__ == "__main__":
    fig = build_figure()

    for ext in ("pdf", "png"):
        out = _OUT / f"Image_1.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(str(out), bbox_inches="tight", dpi=dpi)
        print(f"saved → {out.relative_to(_ROOT)}")

    plt.close(fig)
    print("done.")
