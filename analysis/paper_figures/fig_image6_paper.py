"""
fig_image6_paper.py — Paper figure Image_6: 2 × 3 layout.

Log-volume gap heatmap (layer × normalised token position).
Rows    : LLaMA 3.2 3B  |  Pythia 2.8B
Columns : Refinement  |  Control  |  DiD (Refinement − Control)

Data source: results/cache/D/{model}_eigvals.npz
  ref_eigvals, com_eigvals — shape (n_chains, n_layers, 11, 5 levels, 15 eigvals)
  positions — 11 normalised token positions [0.0, 0.1, …, 1.0]

Outputs
-------
  results/Images/Image_6.pdf
  results/Images/Image_6.png

Usage
-----
  python analysis/paper_figures/fig_image6_paper.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT    = Path(__file__).resolve().parent.parent.parent
_D_CACHE = _ROOT / "results" / "caches" / "position" / "mean"
_OUT     = _ROOT / "results" / "paper_figures"
_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_ROOT))
from analysis.utils import normalised_depth_array, setup_plt

setup_plt()

MODELS = [
    ("llama3.2-3b", "LLaMA 3.2 3B"),
    ("pythia-2.8b", "Pythia 2.8B"),
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def _logvol_from_eigvals(eigvals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log-volume: Σ log(λ) over positive eigenvalues. Last axis = eigvals."""
    safe = np.where(eigvals > eps, eigvals, np.nan)
    return np.nansum(np.log(safe), axis=-1)


# ── Cache loading ─────────────────────────────────────────────────────────────

def _load(model: str) -> dict:
    p = _D_CACHE / f"{model}_eigvals.npz"
    if not p.exists():
        raise FileNotFoundError(f"D cache not found: {p}")
    return dict(np.load(str(p), allow_pickle=True))


def _compute_gaps(model: str):
    """
    Returns (ref_gap, com_gap, nd, positions) each with shape (L, P).
    """
    d         = _load(model)
    ref_eig   = d["ref_eigvals"]   # (n_chains, L, P, 5, E)
    com_eig   = d["com_eigvals"]
    positions = d["positions"]
    L         = int(d["n_layers"])
    nd        = normalised_depth_array(L)

    ref_lv = _logvol_from_eigvals(ref_eig)   # (n_chains, L, P, 5)
    com_lv = _logvol_from_eigvals(com_eig)

    # S0 → S4 gap per chain, then mean across chains
    ref_gap = np.nanmean(ref_lv[..., 4] - ref_lv[..., 0], axis=0)   # (L, P)
    com_gap = np.nanmean(com_lv[..., 4] - com_lv[..., 0], axis=0)

    return ref_gap, com_gap, nd, positions


# ── Heatmap cell ──────────────────────────────────────────────────────────────

def _draw_cell(ax, mat, nd, positions, title, cmap, vmin, vmax, cbar_label):
    im = ax.imshow(
        mat.T,
        origin="lower",
        aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[nd[0], nd[-1], positions[0], positions[-1]],
    )
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Token position", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar_label)


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure() -> plt.Figure:
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8),
        squeeze=False, layout="constrained",
    )

    col_titles = ["Refinement", "Control", "DiD = Refinement − Control"]

    for row_idx, (model_key, model_label) in enumerate(MODELS):
        ref_gap, com_gap, nd, positions = _compute_gaps(model_key)
        did = ref_gap - com_gap

        arm_max = max(float(np.nanmax(np.abs(ref_gap))),
                      float(np.nanmax(np.abs(com_gap))), 1e-9)
        did_max = max(float(np.nanmax(np.abs(did))), 1e-9)

        panels = [
            (ref_gap, "RdBu_r", -arm_max, arm_max, "Δ log-vol"),
            (com_gap, "RdBu_r", -arm_max, arm_max, "Δ log-vol"),
            (did,     "PuOr_r", -did_max, did_max, "Δ log-vol (DiD)"),
        ]

        for col_idx, (mat, cmap, vmin, vmax, cbar_label) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            # Column titles only on top row
            title = col_titles[col_idx] if row_idx == 0 else ""
            _draw_cell(ax, mat, nd, positions, title,
                       cmap, vmin, vmax, cbar_label)

            # x-label on bottom row only
            if row_idx == len(MODELS) - 1:
                ax.set_xlabel("Normalised depth", fontsize=9)

            # y-label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{model_label}\nToken position", fontsize=8)
            else:
                ax.set_ylabel("", fontsize=8)

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig = build_figure()
    for ext in ("pdf", "png"):
        out = _OUT / f"Image_6.{ext}"
        fig.savefig(str(out), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"saved → {out.relative_to(_ROOT)}")
    plt.close(fig)
    print("done.")
