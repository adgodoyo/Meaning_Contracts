"""
fig_image2_paper.py — Paper figure Image_2: 1 × 2 layout.

Left panel  : Gemma 4 (E4B) — Mahalanobis d_M, raw condition
              (top-left panel of results/figures/per_model/gemma4-e4b_Z2_mahalanobis.png)
              Line colour and style: per category (Z2 CAT_STYLE scheme).

Right panel : All-models aggregate — Mahalanobis d_M, raw (no deflation)
              (inspired by left panel of AGG_A2_metric_comparison.png, raw condition)
              Line colour: per category (Z2 CAT_STYLE scheme).
              Line style:  per model (pythia solid, llama dashed, qwen -., gemma :).

Data loaded from existing caches — no recomputation.

Category relabelling (labels only, colours and line styles unchanged):
  Code C1 (dark-red, solid)  →  paper label C2  "Same factor, same language"
  Code C2 (red,      dashed) →  paper label C1  "Exact translation, diff language"
  C3 … C7 unchanged.
  Legend order: C1 (exact translation) first.

Outputs
-------
  results/Images/Image_2.pdf   (vector)
  results/Images/Image_2.png   (300 dpi)

Usage
-----
  python analysis/paper_figures/fig_image2_paper.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT  = Path(__file__).resolve().parent.parent.parent
_Z2_CACHE = _ROOT / "results" / "caches" / "multilingual" / "mean"
_A2_CACHE = _ROOT / "results" / "caches" / "multilingual" / "mean" / "A2"
_OUT   = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import (
    MODELS, model_linestyle, normalised_depth_array, setup_plt,
)

setup_plt()

# ── Category styling (Z2 scheme — same as Image 1) ───────────────────────────
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

# Paper labels: C1 ↔ C2 swapped
CAT_LABEL_PAPER: dict[str, str] = {
    "C1": "C2  Same factor, same language",
    "C2": "C1  Exact translation, diff language",
    "C3": "C3  Same factor, diff language",
    "C4": "C4  Same domain, diff factor, same lang",
    "C5": "C5  Same domain, diff factor, diff lang",
    "C6": "C6  Different domain, same language",
    "C7": "C7  Different domain, diff language",
}

# Legend order: paper C1 first (= code C2), then paper C2 (= code C1), C3–C7
LEGEND_ORDER = ["C2", "C1", "C3", "C4", "C5", "C6", "C7"]

MODEL_LABEL = {
    "pythia-2.8b": "Pythia 2.8B",
    "llama3.2-3b": "LLaMA 3.2 3B",
    "qwen2.5-3b":  "Qwen 2.5 3B",
    "gemma4-e4b":  "Gemma 4 E4B",
}

POOLING = "mean"


# ── Cache loading ─────────────────────────────────────────────────────────────

def _load_z2(model: str) -> dict:
    for p in [
        _Z2_CACHE / f"{model}_{POOLING}_R128.npz",
        _Z2_CACHE / f"{model}_{POOLING}.npz",
    ]:
        if p.exists():
            return dict(np.load(str(p), allow_pickle=True))
    raise FileNotFoundError(f"Z2 cache missing for {model}")


def _load_a2(model: str) -> dict:
    p = _A2_CACHE / f"{model}_{POOLING}.npz"
    if not p.exists():
        raise FileNotFoundError(f"A2 cache missing for {model}")
    return dict(np.load(str(p), allow_pickle=True))


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12 + 2.0, 5))

    # ── Left panel: Gemma Z2 Mahalanobis raw ─────────────────────────────────
    ax_left = axes[0]
    z2_gemma = _load_z2("gemma4-e4b")
    L_gemma  = int(z2_gemma["L"])
    nd_gemma = normalised_depth_array(L_gemma)

    for cat in CATS:
        color, ls, lw = CAT_STYLE[cat]
        ax_left.plot(nd_gemma, z2_gemma[f"mahal_raw_{cat}"],
                     color=color, linestyle=ls, linewidth=lw)

    ax_left.set_xlabel("Normalised depth", fontsize=10)
    ax_left.set_ylabel("Mahalanobis distance", fontsize=10)
    ax_left.set_title("Gemma 4 E4B — Mahalanobis (raw)", fontsize=10)

    # ── Right panel: All-models A2 Mahalanobis residualized ──────────────────
    ax_right = axes[1]

    for model in MODELS:
        a2 = _load_a2(model)
        L  = int(a2["L"])
        nd = normalised_depth_array(L)
        ls = model_linestyle(model)

        for cat in CATS:
            color = CAT_STYLE[cat][0]
            ax_right.plot(nd, a2[f"dm_raw_{cat}"],
                          color=color, linestyle=ls,
                          linewidth=1.1, alpha=0.8)

    ax_right.set_xlabel("Normalised depth", fontsize=10)
    ax_right.set_ylabel("Mahalanobis distance", fontsize=10)
    ax_right.set_title("All models — Mahalanobis (raw)", fontsize=10)

    # ── Shared legend ─────────────────────────────────────────────────────────
    # Categories in paper order
    cat_handles = [
        plt.Line2D([0], [0],
                   color=CAT_STYLE[cat][0],
                   linestyle=CAT_STYLE[cat][1],
                   linewidth=1.6,
                   label=CAT_LABEL_PAPER[cat])
        for cat in LEGEND_ORDER
    ]

    # Separator line (empty label)
    separator = plt.Line2D([0], [0], color="none", label="")

    # Models (line style in neutral gray)
    model_handles = [
        plt.Line2D([0], [0], color="dimgray",
                   linestyle=model_linestyle(m),
                   linewidth=1.6,
                   label=MODEL_LABEL[m])
        for m in MODELS
    ]

    # Vertical legend on the right
    fig.legend(
        handles=cat_handles + [separator] + model_handles,
        fontsize=7.5,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        bbox_transform=fig.transFigure,
        frameon=True,
        borderaxespad=0.2,
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig = build_figure()

    for ext in ("pdf", "png"):
        out = _OUT / f"Image_2.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(str(out), bbox_inches="tight", dpi=dpi)
        print(f"saved → {out.relative_to(_ROOT)}")

    plt.close(fig)
    print("done.")
