"""
make_paper_figures.py -- every figure of the paper, from the caches.

Figures 1-4 reuse the build_figure() functions of fig_image{1,2,3,4}_paper.py
with their shared legend placed below the panels. Figure 5 is a 2 x 2 layout:
rows are the two readouts (mean pooling, last token), columns the two spread
measures (top-5-deflated log-volume, raw participation ratio), each as the gap
from S0, refinement (solid) vs control (dashed), 95% CI, Llama 3.2 3B.
Figure 6 comes from fig_image6_paper.py; Figures 7-8 from
analysis/13_adversarial_profiles.py.

Data sources
------------
  results/caches/multilingual/{mean,last}/        Figures 1-3
  results/caches/refinement/mean/                 Figure 4
  results/caches/refinement/{mean,last}/ZY        Figure 5 (log-volume)
  results/caches/refinement/{mean,last}/B1_pr_cache  Figure 5 (participation ratio), 7, 8
  results/caches/position/mean/                   Figure 6

Output
------
  results/paper_figures/Image_{1,2,3,4,6}.png, Image_5_2x2.png,
  adv_{near_synonym,scope_hedge}_profiles.png

Usage
-----
  python analysis/paper_figures/make_paper_figures.py
"""

from pathlib import Path
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as mfigure
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_OUT  = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import normalised_depth_array, setup_plt


# ── Legend capture: record fig.legend() calls instead of drawing them ─────────

_real_fig_legend = mfigure.Figure.legend


def _capturing_legend(self, *args, **kwargs):
    self._captured_legend = kwargs.get("handles", args[0] if args else None)
    return None


def _load(module_file: str):
    spec = importlib.util.spec_from_file_location(module_file[:-3], _HERE / module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rebuild_with_bottom_legend(module_file: str, extra_width: float,
                                ncol: int) -> plt.Figure:
    """Build one of the base figures and redraw its legend below the panels.

    extra_width : inches the original figure reserved for its right legend.
    """
    mod = _load(module_file)
    mfigure.Figure.legend = _capturing_legend
    try:
        fig = mod.build_figure()
    finally:
        mfigure.Figure.legend = _real_fig_legend

    handles = [h for h in fig._captured_legend if h.get_label()]  # drop spacers
    w, h = fig.get_size_inches()
    fig.set_size_inches(w - extra_width, h)
    n_rows = int(np.ceil(len(handles) / ncol))
    bottom = (0.25 + 0.22 * n_rows) / h
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, bottom, 1, 1])
    return fig


# ── New 2 x 2 Figure 5 ────────────────────────────────────────────────────────

MODEL    = "llama3.2-3b"
N_CHAINS = 20
Z_95     = 1.96
TARGETS  = [1, 2, 3, 4]
LW       = 1.8
ALPHA    = 0.15
_CMAP        = matplotlib.colormaps["viridis"]
LEVEL_COLORS = [mcolors.to_hex(_CMAP(t)) for t in [0.1, 0.3, 0.5, 0.7, 0.9]]

_LV_CACHE = {"mean": _ROOT / "results" / "caches" / "refinement" / "mean" / "ZY",
             "last": _ROOT / "results" / "caches" / "refinement" / "last" / "ZY"}
_PR_CACHE = {"mean": _ROOT / "results" / "caches" / "refinement" / "mean" / "B1_pr_cache",
             "last": _ROOT / "results" / "caches" / "refinement" / "last" / "B1_pr_cache"}


def _lv_gap(pool: str, arm: str):
    """Top-5-deflated log-volume gap from S0; CI from per-chain differences."""
    d = np.load(_LV_CACHE[pool] / f"{MODEL}_{arm}_{pool}_id_lv.npz", allow_pickle=True)
    chain_lv = d["chain_lv"]                      # (chains, L, 5)
    L = chain_lv.shape[1]
    gap, ci = np.zeros((L, 4)), np.zeros((L, 4))
    for i, k in enumerate(TARGETS):
        per_chain = chain_lv[:, :, k] - chain_lv[:, :, 0]
        gap[:, i] = per_chain.mean(axis=0)
        ci[:, i]  = Z_95 * per_chain.std(axis=0, ddof=1) / np.sqrt(per_chain.shape[0])
    return gap, ci


def _pr_gap(pool: str, arm: str):
    """Raw participation-ratio gap from S0; conservative CI (independent SEs)."""
    d = np.load(_PR_CACHE[pool] / f"{MODEL}_{arm}_raw.npz", allow_pickle=True)
    mean_pr, sd_pr = d["mean_pr"], d["sd_pr"]     # (L, 5)
    L = mean_pr.shape[0]
    gap, ci = np.zeros((L, 4)), np.zeros((L, 4))
    for i, k in enumerate(TARGETS):
        gap[:, i] = mean_pr[:, k] - mean_pr[:, 0]
        ci[:, i]  = Z_95 * np.sqrt((sd_pr[:, k] / np.sqrt(N_CHAINS)) ** 2 +
                                   (sd_pr[:, 0] / np.sqrt(N_CHAINS)) ** 2)
    return gap, ci


def _draw_panel(ax, gap_fn, pool, ylabel, title):
    g_ref, c_ref = gap_fn(pool, "refinement")
    g_com, c_com = gap_fn(pool, "commonality")
    nd = normalised_depth_array(g_ref.shape[0])
    for i, k in enumerate(TARGETS):
        col = LEVEL_COLORS[k]
        ax.plot(nd, g_ref[:, i], color=col, linestyle="-", linewidth=LW)
        ax.fill_between(nd, g_ref[:, i] - c_ref[:, i], g_ref[:, i] + c_ref[:, i],
                        color=col, alpha=ALPHA)
        ax.plot(nd, g_com[:, i], color=col, linestyle="--", linewidth=LW)
        ax.fill_between(nd, g_com[:, i] - c_com[:, i], g_com[:, i] + c_com[:, i],
                        color=col, alpha=ALPHA * 0.6)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)


def build_figure5_2x2() -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=True)
    rows = [("mean", "Mean pooling (sentence-level)"),
            ("last", "Last token (token-local)")]
    for r, (pool, row_label) in enumerate(rows):
        _draw_panel(axes[r, 0], _lv_gap, pool, "Δ Log-volume (top-5 deflated)",
                    f"{row_label}: log-volume")
        _draw_panel(axes[r, 1], _pr_gap, pool, "Δ Participation ratio (raw)",
                    f"{row_label}: participation ratio")
    for ax in axes[1]:
        ax.set_xlabel("Normalised depth  ℓ / (L−1)", fontsize=10)

    handles = [plt.Line2D([0], [0], color=LEVEL_COLORS[k], linewidth=2,
                          label=f"S0 → S{k}") for k in TARGETS]
    handles += [
        plt.Line2D([0], [0], color="dimgray", linestyle="-", linewidth=2, label="Refinement"),
        plt.Line2D([0], [0], color="dimgray", linestyle="--", linewidth=2, label="Control"),
        mpatches.Patch(facecolor="dimgray", alpha=0.20, label="95% CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def build_adversarial_profiles(adv_arm: str) -> plt.Figure:
    """Appendix adversarial-profile figure (final_codebase script 13), with the
    control arm labelled "Control" to match the paper's terminology."""
    spec = importlib.util.spec_from_file_location(
        "adv_profiles", _ROOT / "analysis" / "13_adversarial_profiles.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ARM_LABELS = {**mod.ARM_LABELS, mod.ARM_COMMONALITY: "Control"}
    fig = mod.build_figure(adv_arm, "mean")
    fig.legends[0].set_bbox_to_anchor((0.5, -0.04), transform=fig.transFigure)  # clear x-labels
    return fig


def _save(fig, stem):
    out = _OUT / f"{stem}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=300)
    print(f"saved → {out.relative_to(_ROOT)}")
    plt.close(fig)


if __name__ == "__main__":
    # (module, inches reserved for the right legend in the original, legend columns)
    specs = [("fig_image1_paper.py", 1.5, 4, "Image_1"),
             ("fig_image2_paper.py", 2.0, 4, "Image_2"),
             ("fig_image3_paper.py", 2.0, 4, "Image_3"),
             ("fig_image4_paper.py", 2.0, 7, "Image_4")]
    for module_file, extra, ncol, stem in specs:
        _save(_rebuild_with_bottom_legend(module_file, extra, ncol), stem)
    setup_plt()
    _save(build_figure5_2x2(), "Image_5_2x2")
    for adv_arm in ("adv_near_synonym", "adv_scope_hedge"):
        _save(build_adversarial_profiles(adv_arm), f"{adv_arm}_profiles")
    # Figure 6 has no shared legend to move.
    _save(_load("fig_image6_paper.py").build_figure(), "Image_6")
    print("done.")
