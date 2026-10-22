"""
D_position_heatmaps.py — PR-gap and log-volume-gap heatmaps in (layer × position).

Reads cached per-(arm, chain, level, layer, position) covariance eigenvalues
produced by D_extract_per_position.py, derives per-chain Participation Ratio
and log-volume, computes the S0→S4 gap per arm and the diff-in-differences
(refinement − control), and renders three-panel heatmaps.

Outputs (saved to figures/D/ or figures_last/D/)
-------
  {model}_D1_PR_gap_heatmap.png       — refinement | control | DiD
  {model}_D2_logvol_gap_heatmap.png   — refinement | control | DiD

Usage
-----
  python analysis/05_position_analysis.py
  python analysis/05_position_analysis.py --models pythia-2.8b
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils import (
    MODELS, TABLES_DIR,
    get_cache_dir, get_figures_dir,
    normalised_depth_array, setup_plt, SEED,
)

setup_plt()
np.random.seed(SEED)

_D_CACHE = None  # set in _load_eigvals


def _get_d_cache() -> Path:
    return get_cache_dir("position", "mean")


# ── Output directory ─────────────────────────────────────────────────────────

def _d_dir() -> Path:
    d = get_figures_dir("mean", "position_analysis")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str, model: str = "") -> None:
    if model:
        out = _d_dir() / f"{model}_{name}.png"
    else:
        out = _d_dir() / f"{name}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as _plt2; _plt2.close(fig)
    print(f"  saved → {out.relative_to(Path(__file__).resolve().parents[1])}", flush=True)


# ── Metric derivation from eigenvalues ───────────────────────────────────────

def _pr_from_eigvals(eigvals: np.ndarray) -> np.ndarray:
    """
    Participation ratio: PR = (Σλ)² / Σλ² along the last axis.
    eigvals shape: (..., n_eigvals).  Returns shape (...,).
    """
    eigvals = np.maximum(eigvals, 0.0)
    num = eigvals.sum(axis=-1) ** 2
    den = (eigvals ** 2).sum(axis=-1)
    return np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)


def _logvol_from_eigvals(eigvals: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Log-volume: Σ log(λ) over positive eigenvalues only (drops zero/negative).
    """
    safe = np.where(eigvals > eps, eigvals, np.nan)
    return np.nansum(np.log(safe), axis=-1)


# ── Cache loader ─────────────────────────────────────────────────────────────

def _load_eigvals(model: str) -> dict | None:
    p = _get_d_cache() / f"{model}_eigvals.npz"
    if not p.exists():
        print(f"  [{model}] no D cache at {p} — run "
              f"D_extract_per_position.py first.", flush=True)
        return None
    d = dict(np.load(str(p), allow_pickle=True))
    return d


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def _heatmap(ax, mat: np.ndarray, nd: np.ndarray, positions: np.ndarray,
             title: str, cmap: str, vmin: float, vmax: float,
             cbar_label: str) -> None:
    """
    mat: shape (n_layers, n_positions). x-axis = layer (depth), y-axis = position.
    """
    # Plot with imshow; pass extent so axes show actual values.
    im = ax.imshow(
        mat.T,
        origin="lower",
        aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[nd[0], nd[-1], positions[0], positions[-1]],
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Normalised depth (layer)")
    ax.set_ylabel("Normalised token position")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=cbar_label)


def _plot_metric_heatmap(model: str, metric_name: str,
                         ref_gap: np.ndarray, com_gap: np.ndarray,
                         nd: np.ndarray, positions: np.ndarray,
                         filename: str) -> None:
    """
    Three-panel heatmap: Refinement | Control | DiD.
    ref_gap, com_gap shape: (n_layers, n_positions).
    """
    did = ref_gap - com_gap

    # Symmetric color limits for each panel using its own data range
    ref_max = float(np.nanmax(np.abs(ref_gap)))
    com_max = float(np.nanmax(np.abs(com_gap)))
    arm_max = max(ref_max, com_max, 1e-9)
    did_max = max(float(np.nanmax(np.abs(did))), 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"{model}  —  {metric_name} gap (S4 − S0) heatmap "
        "(layer × normalised token position)",
        fontsize=11, fontweight="bold",
    )

    _heatmap(axes[0], ref_gap, nd, positions,
             title="Refinement", cmap="RdBu_r",
             vmin=-arm_max, vmax=arm_max,
             cbar_label=f"Δ {metric_name}")
    _heatmap(axes[1], com_gap, nd, positions,
             title="Control", cmap="RdBu_r",
             vmin=-arm_max, vmax=arm_max,
             cbar_label=f"Δ {metric_name}")
    _heatmap(axes[2], did, nd, positions,
             title="DiD = Refinement − Control", cmap="PuOr_r",
             vmin=-did_max, vmax=did_max,
             cbar_label=f"Δ {metric_name}")

    fig.subplots_adjust(top=0.90, wspace=0.35)
    _save(fig, filename, model=model)


# ── Table helpers ─────────────────────────────────────────────────────────────

def _peak_row(mat: np.ndarray, nd: np.ndarray, positions: np.ndarray) -> dict:
    idx = np.unravel_index(np.nanargmax(mat), mat.shape)
    return {
        "peak_value":  float(mat[idx]),
        "at_depth_nd": float(nd[idx[0]]),
        "at_position": float(positions[idx[1]]),
        "mean_value":  float(np.nanmean(mat)),
    }


def _save_tables(
    model: str,
    nd: np.ndarray,
    positions: np.ndarray,
    ref_pr: np.ndarray, com_pr: np.ndarray,
    ref_lv: np.ndarray, com_lv: np.ndarray,
) -> None:
    """
    Three CSV tables (appended, one row per model):

    D_peak.csv        — peak value, location, and mean for each metric × arm.
    D_by_position.csv — metric averaged over layers, one column per position.
    D_by_depth.csv    — metric averaged over positions, for early/mid/late bands.
    """
    did_pr = ref_pr - com_pr
    did_lv = ref_lv - com_lv
    P      = len(positions)

    bands = [
        ("early (ND<0.33)",      nd <  0.33),
        ("mid (0.33≤ND<0.67)",  (nd >= 0.33) & (nd < 0.67)),
        ("late (ND≥0.67)",       nd >= 0.67),
    ]
    pos_labels   = [f"p={p:.1f}" for p in positions]
    band_labels  = [b[0] for b in bands]

    combos = [
        ("PR",      ref_pr, com_pr, did_pr),
        ("log-vol", ref_lv, com_lv, did_lv),
    ]
    arms = [("refinement", 0), ("control", 1), ("DiD", 2)]

    # ── D_peak.csv ────────────────────────────────────────────────────────────
    peak_path   = TABLES_DIR / "D_peak.csv"
    peak_fields = ["model", "metric", "arm",
                   "peak_value", "at_depth_nd", "at_position", "mean_value"]
    peak_rows   = []
    for metric, ref_m, com_m, did_m in combos:
        for arm_name, idx in arms:
            mat = [ref_m, com_m, did_m][idx]
            r   = _peak_row(mat, nd, positions)
            peak_rows.append({
                "model": model, "metric": metric, "arm": arm_name,
                "peak_value":  f"{r['peak_value']:.4f}",
                "at_depth_nd": f"{r['at_depth_nd']:.3f}",
                "at_position": f"{r['at_position']:.1f}",
                "mean_value":  f"{r['mean_value']:.4f}",
            })
    _write_csv(peak_path, peak_fields, peak_rows)

    # ── D_by_position.csv ─────────────────────────────────────────────────────
    pos_path   = TABLES_DIR / "D_by_position.csv"
    pos_fields = ["model", "metric", "arm"] + pos_labels
    pos_rows   = []
    for metric, ref_m, com_m, did_m in combos:
        for arm_name, idx in arms:
            mat  = [ref_m, com_m, did_m][idx]
            vals = np.nanmean(mat, axis=0)     # (P,) averaged over layers
            row  = {"model": model, "metric": metric, "arm": arm_name}
            row.update({pos_labels[pi]: f"{vals[pi]:.4f}" for pi in range(P)})
            pos_rows.append(row)
    _write_csv(pos_path, pos_fields, pos_rows)

    # ── D_by_depth.csv ────────────────────────────────────────────────────────
    dep_path   = TABLES_DIR / "D_by_depth.csv"
    dep_fields = ["model", "metric", "arm"] + band_labels
    dep_rows   = []
    for metric, ref_m, com_m, did_m in combos:
        for arm_name, idx in arms:
            mat              = [ref_m, com_m, did_m][idx]
            per_layer        = np.nanmean(mat, axis=1)   # (L,) averaged over positions
            row              = {"model": model, "metric": metric, "arm": arm_name}
            for label, mask in bands:
                row[label] = f"{np.nanmean(per_layer[mask]):.4f}"
            dep_rows.append(row)
    _write_csv(dep_path, dep_fields, dep_rows)

    print(f"  [{model}] tables saved → D_peak / D_by_position / D_by_depth",
          flush=True)


def _write_csv(path: Path, fields: list, rows: list) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ── Per-model runner ──────────────────────────────────────────────────────────

def run(model: str) -> None:
    print(f"\n{'='*60}\nD-heatmaps — {model}\n{'='*60}", flush=True)
    cached = _load_eigvals(model)
    if cached is None:
        return

    # eigvals shape: (n_chains, n_layers, n_positions, 5 levels, n_eigvals)
    ref_eig   = cached["ref_eigvals"]
    com_eig   = cached["com_eigvals"]
    positions = cached["positions"]
    L         = int(cached["n_layers"])
    nd        = normalised_depth_array(L)

    # Derive PR and log-vol per (chain, layer, position, level)
    ref_pr = _pr_from_eigvals(ref_eig)         # (n_ref, L, P, 5)
    com_pr = _pr_from_eigvals(com_eig)         # (n_com, L, P, 5)
    ref_lv = _logvol_from_eigvals(ref_eig)
    com_lv = _logvol_from_eigvals(com_eig)

    # Per-chain S0→S4 gap
    ref_pr_gap = ref_pr[..., 4] - ref_pr[..., 0]    # (n_ref, L, P)
    com_pr_gap = com_pr[..., 4] - com_pr[..., 0]
    ref_lv_gap = ref_lv[..., 4] - ref_lv[..., 0]
    com_lv_gap = com_lv[..., 4] - com_lv[..., 0]

    # Mean across chains → (L, P)
    ref_pr_gap_mean = np.nanmean(ref_pr_gap, axis=0)
    com_pr_gap_mean = np.nanmean(com_pr_gap, axis=0)
    ref_lv_gap_mean = np.nanmean(ref_lv_gap, axis=0)
    com_lv_gap_mean = np.nanmean(com_lv_gap, axis=0)

    _plot_metric_heatmap(
        model, "PR",
        ref_pr_gap_mean, com_pr_gap_mean,
        nd, positions,
        "D1_PR_gap_heatmap",
    )
    _plot_metric_heatmap(
        model, "log-vol",
        ref_lv_gap_mean, com_lv_gap_mean,
        nd, positions,
        "D2_logvol_gap_heatmap",
    )
    _save_tables(
        model, nd, positions,
        ref_pr_gap_mean, com_pr_gap_mean,
        ref_lv_gap_mean, com_lv_gap_mean,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D: PR/log-vol gap heatmaps (layer × position)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    for model in args.models:
        run(model)

    print("\nD_position_heatmaps complete.", flush=True)
