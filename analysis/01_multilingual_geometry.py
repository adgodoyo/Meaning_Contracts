"""
01_multilingual_geometry.py — Per-model deflation grid figures for 3 metrics × 12 conditions.

Conditions (12 total, 4 rows × 3 cols)
---------------------------------------
  row 0: raw        | top-1    | top-2
  row 1: top-3      | top-4    | top-5
  row 2: top-6      | top-7    | top-8
  row 3: top-9      | top-10   | lang-resid

Metrics
-------
  Cosine similarity, Mahalanobis distance, Euclidean distance

Outputs (per model)
-------------------
  results/figures/per_model/{model}_Z2_cosine.png
  results/figures/per_model/{model}_Z2_mahalanobis.png
  results/figures/per_model/{model}_Z2_euclidean.png
  ... plus _zoom variants for each

Data sources
------------
  Cosine : computed inline (or loaded from Z2 cache)
           Falls back to A1 cache (raw/lang-resid) + A3 cache (top-K) if available.
  Mahal  : computed inline, cached to results/cache/Z2/{model}_{pooling}.npz
  Eucl   : same Z2 cache

Run order: Independent — loads activations on first run, uses Z2 cache on re-runs.

Usage
-----
  python analysis/01_multilingual_geometry.py
  python analysis/01_multilingual_geometry.py --models llama3.2-3b
  python analysis/01_multilingual_geometry.py --force
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, PAIR_CATEGORIES, get_cache_dir, get_figures_dir,
    load_activations, load_multilingual_labels,
    build_multilingual_pair_indices, pairs_to_array,
    language_residualization, top_k_deflation,
    normalised_depth_array,
    category_distances,
    save_figure, setup_plt, SEED,
)

np.random.seed(SEED)
setup_plt()

CATS    = list(PAIR_CATEGORIES.keys())
K_SWEEP = list(range(1, 11))
RANK    = 128   # default; override via --rank (use 256 for last-token pooling)

# ── Color + line-style convention (copied exactly from X_mahal_deflation.py) ──
CAT_STYLE = {
    "C1": ("#c0392b", "-",   1.8),
    "C2": ("#e74c3c", "--",  2.2),
    "C3": ("#e67e22", "--",  1.8),
    "C4": ("#27ae60", "-",   1.8),
    "C5": ("#1abc9c", "--",  1.8),
    "C6": ("#2980b9", "-",   1.8),
    "C7": ("#154360", "--",  1.8),
}

CAT_LABELS = {
    "C1": "C1  Same factor, same lang",
    "C2": "C2  Exact translation, diff lang",
    "C3": "C3  Same factor, diff lang",
    "C4": "C4  Same domain, diff fac, same lang",
    "C5": "C5  Same domain, diff fac, diff lang",
    "C6": "C6  Diff topic, same lang",
    "C7": "C7  Diff topic, diff lang",
}

# 12 conditions in 4 × 3 grid order
_GRID_CONDITIONS = (
    ["raw",    "top1",  "top2"],
    ["top3",   "top4",  "top5"],
    ["top6",   "top7",  "top8"],
    ["top9",   "top10", "lang_resid"],
)
_GRID_LABELS = {
    "raw":        "Raw",
    "lang_resid": "Lang-resid",
    **{f"top{k}": f"Top-{k}" for k in range(1, 11)},
}


def _get_z2_cache(pooling: str) -> Path:
    d = get_cache_dir("multilingual", pooling)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _z2_cache_path(model: str, pooling: str, rank: int) -> Path:
    return _get_z2_cache(pooling) / f"{model}_{pooling}_R{rank}.npz"


def _save_z2_cache(model: str, pooling: str, rank: int, data: dict) -> None:
    p = _z2_cache_path(model, pooling, rank)
    np.savez(str(p), **data)
    print(f"  [{model}] Z2 cache saved → {p.relative_to(Path(__file__).resolve().parents[1])}")


def _load_z2_cache(model: str, pooling: str, rank: int) -> dict | None:
    p = _z2_cache_path(model, pooling, rank)
    if not p.exists():
        # Fallback: try legacy cache without rank suffix (rank=128 runs pre-dating this change)
        legacy = _get_z2_cache(pooling) / f"{model}_{pooling}.npz"
        if rank == 128 and legacy.exists():
            d = dict(np.load(str(legacy), allow_pickle=True))
            print(f"  [{model}] Z2 loaded from legacy cache (R=128).", flush=True)
            return d
        return None
    d = dict(np.load(str(p), allow_pickle=True))
    print(f"  [{model}] Z2 loaded from cache (R={rank}).", flush=True)
    return d


def _try_load_a1(model: str, pooling: str) -> dict | None:
    return None


def _try_load_a3(model: str, pooling: str) -> dict | None:
    return None


# ── Mahalanobis + Euclidean at one layer ──────────────────────────────────────

def _compute_mahal_eucl(X: np.ndarray, pairs: dict, rank: int = RANK) -> tuple[dict, dict]:
    """
    Compute mean Mahalanobis and Euclidean distances for all 7 categories.
    Uses truncated SVD at the given rank for the Mahalanobis metric.
    Returns (mahal_means, eucl_means) each {cat: float}.
    """
    X64 = X.astype(np.float64)
    Xc  = X64 - X64.mean(axis=0)
    N, D = Xc.shape
    k    = min(rank, N - 1, D - 1)

    np.random.seed(SEED)
    _, s, Vt     = randomized_svd(Xc, n_components=k, random_state=SEED)
    eigvals      = (s ** 2) / max(N - 1, 1)
    inv_sqrt_eig = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))

    X_pc = Xc @ Vt.T   # (N, k)

    mahal: dict[str, float] = {}
    eucl:  dict[str, float] = {}

    for c in CATS:
        p = pairs[c]
        if len(p) == 0:
            mahal[c] = np.nan
            eucl[c]  = np.nan
            continue
        diff_pc  = X_pc[p[:, 0]] - X_pc[p[:, 1]]    # (P, k)
        diff_raw = Xc[p[:, 0]]   - Xc[p[:, 1]]       # (P, D)
        mahal[c] = float(np.linalg.norm(diff_pc * inv_sqrt_eig[None, :], axis=1).mean())
        eucl[c]  = float(np.linalg.norm(diff_raw, axis=1).mean())

    return mahal, eucl


# ── Per-model computation ──────────────────────────────────────────────────────

def run(model: str, pooling: str = "mean", force: bool = False,
        rank: int = RANK) -> dict:
    print(f"\n{'='*60}\nZ2 — {model}\n{'='*60}")

    CONDITIONS = ["raw"] + [f"top{k}" for k in K_SWEEP] + ["lang_resid"]

    # ── Try loading from Z2 cache first ──────────────────────────────────────
    cached = None if force else _load_z2_cache(model, pooling, rank)
    if cached is not None:
        L  = int(cached["L"])
        nd = normalised_depth_array(L)

        profiles_cos   = {cond: {c: cached[f"cos_{cond}_{c}"]   for c in CATS} for cond in CONDITIONS}
        profiles_mahal = {cond: {c: cached[f"mahal_{cond}_{c}"] for c in CATS} for cond in CONDITIONS}
        profiles_eucl  = {cond: {c: cached[f"eucl_{cond}_{c}"]  for c in CATS} for cond in CONDITIONS}
        return {"model": model, "L": L, "nd": nd,
                "profiles_cos":   profiles_cos,
                "profiles_mahal": profiles_mahal,
                "profiles_eucl":  profiles_eucl}

    # ── Load activations ──────────────────────────────────────────────────────
    acts    = load_activations(model, "multilingual", pooling)
    N, L, D = acts.shape
    nd      = normalised_depth_array(L)
    labels  = load_multilingual_labels()

    pair_lists = build_multilingual_pair_indices(labels)
    pairs      = {c: pairs_to_array(pair_lists[c]) for c in CATS}

    # ── Try A1 / A3 caches for cosine (skip recompute if available) ───────────
    a1 = _try_load_a1(model, pooling)
    a3 = _try_load_a3(model, pooling)

    profiles_cos   = {cond: {c: np.empty(L) for c in CATS} for cond in CONDITIONS}
    profiles_mahal = {cond: {c: np.empty(L) for c in CATS} for cond in CONDITIONS}
    profiles_eucl  = {cond: {c: np.empty(L) for c in CATS} for cond in CONDITIONS}

    need_cos_raw       = (a1 is None)
    need_cos_langresid = (a1 is None)
    need_cos_topk      = {k: (a3 is None) for k in K_SWEEP}

    if a1 is not None:
        for c in CATS:
            profiles_cos["raw"][c]        = a1[f"raw_mean_{c}"]
            profiles_cos["lang_resid"][c] = a1[f"res_mean_{c}"]
        print(f"  [{model}] cosine raw + lang-resid loaded from A1 cache.", flush=True)

    if a3 is not None:
        for k in K_SWEEP:
            for c in CATS:
                profiles_cos[f"top{k}"][c] = a3[f"profile_K{k}_{c}"]
        print(f"  [{model}] cosine top-K deflated loaded from A3 cache.", flush=True)

    # ── Layer loop ────────────────────────────────────────────────────────────
    print(f"  Computing Mahalanobis + Euclidean across {L} layers × 12 conditions...")
    for l in range(L):
        if l % 8 == 0:
            print(f"  layer {l}/{L}...", end=" ", flush=True)

        X_raw = acts[:, l, :].astype(np.float32)
        X_res = language_residualization(X_raw, labels["language"]).astype(np.float32)

        # Cosine — raw
        if need_cos_raw:
            m, _ = category_distances(X_raw, pairs, mode="cosine")
            for c in CATS:
                profiles_cos["raw"][c][l] = m[c]

        # Cosine + Mahal + Eucl — raw
        mh, eu = _compute_mahal_eucl(X_raw, pairs, rank)
        for c in CATS:
            profiles_mahal["raw"][c][l] = mh[c]
            profiles_eucl["raw"][c][l]  = eu[c]

        # Cosine — lang_resid
        if need_cos_langresid:
            mc, _ = category_distances(X_res, pairs, mode="cosine")
            for c in CATS:
                profiles_cos["lang_resid"][c][l] = mc[c]

        # Mahal + Eucl — lang_resid
        mh, eu = _compute_mahal_eucl(X_res, pairs, rank)
        for c in CATS:
            profiles_mahal["lang_resid"][c][l] = mh[c]
            profiles_eucl["lang_resid"][c][l]  = eu[c]

        # top-K conditions
        for k in K_SWEEP:
            cond  = f"top{k}"
            X_def = top_k_deflation(X_raw, k).astype(np.float32)

            if need_cos_topk[k]:
                mc, _ = category_distances(X_def, pairs, mode="cosine")
                for c in CATS:
                    profiles_cos[cond][c][l] = mc[c]

            mh, eu = _compute_mahal_eucl(X_def, pairs, rank)
            for c in CATS:
                profiles_mahal[cond][c][l] = mh[c]
                profiles_eucl[cond][c][l]  = eu[c]

    print()

    # ── Save Z2 cache ─────────────────────────────────────────────────────────
    cache_dict: dict = {"L": L}
    for cond in CONDITIONS:
        for c in CATS:
            cache_dict[f"cos_{cond}_{c}"]   = profiles_cos[cond][c]
            cache_dict[f"mahal_{cond}_{c}"] = profiles_mahal[cond][c]
            cache_dict[f"eucl_{cond}_{c}"]  = profiles_eucl[cond][c]
    _save_z2_cache(model, pooling, rank, cache_dict)

    return {
        "model": model, "L": L, "nd": nd,
        "profiles_cos":   profiles_cos,
        "profiles_mahal": profiles_mahal,
        "profiles_eucl":  profiles_eucl,
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _panel_ylims(data: np.ndarray) -> tuple[float, float]:
    """[data_min − 5% span, data_max + 5% span] ignoring NaN."""
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    span = max(vmax - vmin, 1e-6)
    return vmin - 0.05 * span, vmax + 0.05 * span


def _zoom_ylims(nd: np.ndarray, data: np.ndarray,
                x_lo: float = 0.03, x_hi: float = 0.97) -> tuple[float, float]:
    """
    Y-limits based on 5th–95th percentile of data within [x_lo, x_hi] depth,
    plus 10% margin. Guards against all-NaN slices.
    """
    mask = (nd >= x_lo) & (nd <= x_hi)
    sub  = data[mask]
    sub  = sub[np.isfinite(sub)]
    if len(sub) == 0:
        return _panel_ylims(data)
    p5, p95 = float(np.percentile(sub, 5)), float(np.percentile(sub, 95))
    span = max(p95 - p5, 1e-6)
    return p5 - 0.10 * span, p95 + 0.10 * span


def _plot_metric_grid(
    nd:           np.ndarray,
    L:            int,
    profiles:     dict,
    metric_label: str,
    script_id:    str,
    model:        str,
    pooling:      str,
    zoom:         bool = False,
) -> None:
    """
    4 rows × 3 cols figure for one metric, 12 conditions.
    sharey=False — every panel has its own independent y-limits.
    No stable band shading.
    """
    fig, axes = plt.subplots(
        4, 3,
        figsize=(15, 18),
        sharex=True, sharey=False,
        squeeze=False,
    )
    zoom_tag = " (zoomed)" if zoom else ""
    fig.suptitle(
        f"{model}  —  {metric_label} across 12 deflation conditions{zoom_tag}\n"
        "Color: warm=same-factor | green=same-domain | blue=diff-domain   "
        "Line: solid=same-lang | dashed=diff-lang",
        fontsize=9, fontweight="bold",
    )

    for row_i, row_conds in enumerate(_GRID_CONDITIONS):
        for col_i, cond in enumerate(row_conds):
            ax   = axes[row_i, col_i]
            data_all = []

            for c in CATS:
                color, ls, lw = CAT_STYLE[c]
                ax.plot(
                    nd, profiles[cond][c],
                    color=color, linestyle=ls, linewidth=lw,
                    label=CAT_LABELS[c] if (row_i == 0 and col_i == 0) else None,
                )
                data_all.append(profiles[cond][c])

            all_vals = np.concatenate(data_all)

            if zoom:
                nd_tiled = np.tile(nd, len(data_all))
                ylo, yhi = _zoom_ylims(nd_tiled, all_vals)
                ax.set_xlim(0.03, 0.97)
            else:
                ylo, yhi = _panel_ylims(all_vals)

            ax.set_ylim(ylo, yhi)
            ax.set_title(_GRID_LABELS[cond], fontsize=8)
            if col_i == 0:
                ax.set_ylabel(metric_label, fontsize=7)
            if row_i == 3:
                ax.set_xlabel("Norm. depth", fontsize=7)

    axes[0, 0].legend(fontsize=5.5, ncol=1, loc="best")

    suffix = "_zoom" if zoom else ""
    save_figure(fig, pooling, "multilingual_deflation", f"{model}_{script_id}{suffix}")


def plot_model(res: dict, pooling: str = "mean") -> None:
    model = res["model"]
    nd    = res["nd"]
    L     = res["L"]

    for zoom in (False, True):
        _plot_metric_grid(
            nd, L,
            profiles=res["profiles_cos"],
            metric_label="Cosine similarity",
            script_id="Z2_cosine",
            model=model,
            pooling=pooling,
            zoom=zoom,
        )
        _plot_metric_grid(
            nd, L,
            profiles=res["profiles_mahal"],
            metric_label="Mahalanobis d_M",
            script_id="Z2_mahalanobis",
            model=model,
            pooling=pooling,
            zoom=zoom,
        )
        _plot_metric_grid(
            nd, L,
            profiles=res["profiles_eucl"],
            metric_label="Euclidean d_E",
            script_id="Z2_euclidean",
            model=model,
            pooling=pooling,
            zoom=zoom,
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Z2: per-model deflation grid figures for 3 metrics × 12 conditions")
    parser.add_argument("--models",  nargs="+", default=MODELS)
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    parser.add_argument("--rank",    type=int, default=128,
                        help="Truncation rank for Mahalanobis SVD (default 128; "
                             "use 256 for last-token pooling)")
    parser.add_argument("--force",   action="store_true",
                        help="Recompute even if Z2 cache exists")
    args = parser.parse_args()

    for model in args.models:
        res = run(model, args.pooling, force=args.force, rank=args.rank)
        plot_model(res, args.pooling)

    print("\nZ2_deflation_grids complete.")
