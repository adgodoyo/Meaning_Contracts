"""
fig_image3_paper.py — Paper figure Image_3: 3 × 2 layout.

Rows    : Euclidean raw | Euclidean top-10 (mean pooling) | Euclidean top-10 (last-token)
Columns : Gemma 4 E4B  | Pythia 2.8B

Y-axis:
  Row 0 (raw)          : auto per panel
  Rows 1-2 (deflated)  : fixed [0, 50] for both columns

Data:
  Mean-pooling rows : results/cache/Z2/{model}_mean.npz   (keys eucl_raw_*, eucl_top10_*)
  Last-token row    : results/cache_last/Z2/{model}_last_R256.npz (key eucl_top10_*)

Same Z2 colour scheme and C1 ↔ C2 paper relabelling as Image_1 and Image_2.

Outputs
-------
  results/Images/Image_3.pdf
  results/Images/Image_3.png

Usage
-----
  python analysis/paper_figures/fig_image3_paper.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT      = Path(__file__).resolve().parent.parent.parent
_Z2_MEAN   = _ROOT / "results" / "caches" / "multilingual" / "mean"
_Z2_LAST   = _ROOT / "results" / "caches" / "multilingual" / "last"
_OUT       = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import normalised_depth_array, setup_plt

setup_plt()

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

ROWS = [
    ("gemma4-e4b", "Gemma 4 E4B"),
    ("qwen2.5-3b", "Qwen 2.5 3B"),
]

# ylim: single tuple applied to all rows, or dict {model_key: tuple} for per-row limits
COLUMNS = [
    ("raw_mean",   "Euclidean — raw",                None),
    ("top10_mean", "Euclidean — top-10 (mean pool)",  {"gemma4-e4b": (0, 60), "qwen2.5-3b": (20, 40)}),
    ("top10_last", "Euclidean — top-10 (last token)", {"gemma4-e4b": (0, 60), "qwen2.5-3b": (20, 40)}),
]


# ── Cache loading ─────────────────────────────────────────────────────────────

def _load_mean(model: str) -> dict:
    for p in [
        _Z2_MEAN / f"{model}_mean_R128.npz",
        _Z2_MEAN / f"{model}_mean.npz",
    ]:
        if p.exists():
            return dict(np.load(str(p), allow_pickle=True))
    raise FileNotFoundError(f"Z2 mean cache missing for {model}")


def _load_last(model: str) -> dict:
    p = _Z2_LAST / f"{model}_last_R256.npz"
    if not p.exists():
        raise FileNotFoundError(f"Z2 last-token cache missing for {model}. "
                                f"Run: POOLING_TAG=last python analysis/Z2_deflation_grids.py "
                                f"--models {model} --pooling last --rank 256")
    return dict(np.load(str(p), allow_pickle=True))


def _get_profile(model: str, row_key: str, cat: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (nd, values) for the given model / row / category."""
    if row_key == "raw_mean":
        d  = _load_mean(model)
        nd = normalised_depth_array(int(d["L"]))
        return nd, d[f"eucl_raw_{cat}"]
    elif row_key == "top10_mean":
        d  = _load_mean(model)
        nd = normalised_depth_array(int(d["L"]))
        return nd, d[f"eucl_top10_{cat}"]
    elif row_key == "top10_last":
        d  = _load_last(model)
        nd = normalised_depth_array(int(d["L"]))
        return nd, d[f"eucl_top10_{cat}"]
    else:
        raise ValueError(f"Unknown row_key: {row_key}")


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure() -> plt.Figure:
    n_rows, n_cols = len(ROWS), len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols + 2.0, 4.0 * n_rows),
                             sharex=False, sharey=False,
                             squeeze=False)

    for row_idx, (model_key, model_label) in enumerate(ROWS):
        for col_idx, (col_key, col_title, ylim) in enumerate(COLUMNS):
            ax = axes[row_idx, col_idx]

            for cat in CATS:
                color, ls, lw = CAT_STYLE[cat]
                nd, vals = _get_profile(model_key, col_key, cat)
                ax.plot(nd, vals, color=color, linestyle=ls, linewidth=lw)

            # Fixed y-axis for deflated columns (supports per-row dict)
            if ylim is not None:
                lim = ylim[model_key] if isinstance(ylim, dict) else ylim
                ax.set_ylim(*lim)

            # Row label on left column only
            if col_idx == 0:
                ax.set_ylabel(model_label, fontsize=10, fontweight="bold")

            # Column title on top row only
            if row_idx == 0:
                ax.set_title(col_title, fontsize=9)

            # x-axis label on bottom row only
            if row_idx == n_rows - 1:
                ax.set_xlabel("Normalised depth", fontsize=9)

    # ── Shared legend on the right ───────────────────────────────────────────
    cat_handles = [
        plt.Line2D([0], [0],
                   color=CAT_STYLE[cat][0],
                   linestyle=CAT_STYLE[cat][1],
                   linewidth=1.6,
                   label=CAT_LABEL_PAPER[cat])
        for cat in LEGEND_ORDER
    ]

    fig.legend(handles=cat_handles,
               fontsize=7.5, ncol=1, loc="center left",
               bbox_to_anchor=(0.86, 0.5), bbox_transform=fig.transFigure,
               frameon=True, borderaxespad=0.2)

    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig = build_figure()
    for ext in ("pdf", "png"):
        out = _OUT / f"Image_3.{ext}"
        fig.savefig(str(out), bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"saved → {out.relative_to(_ROOT)}")
    plt.close(fig)
    print("done.")
