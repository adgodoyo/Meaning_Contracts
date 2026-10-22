"""
02_refinement_distances.py — Level-to-level distances in the constraints dataset.

Computes Euclidean, Mahalanobis, and Cosine distances between per-chain mean
representations across constraint levels (S0–S4), contrasting refinement vs
commonality with 95% CI.

Three figures:

  C1_gap.png   (5 × 3)
      Row i = level Si as reference. Each panel: distance from Si to every
      other Sk (k≠i) — 4 pairs of curves (refinement solid, control dashed)
      with CI bands. Columns: Euclidean | Mahalanobis | Cosine.

  C5_layer_drift_did.png   (5 × 3)
      Diff-in-differences of layer drift (refinement − control). Each row =
      one level, each column = one metric. One curve per panel with
      propagated 95% CI; isolates refinement-specific drift from the shared
      layer-by-layer motion baseline.

  C6_drift_did_overlay.png   (1 × 3)
      All five DiD curves overlaid in one panel per metric. Shows which
      level has the largest refinement−control gap at each layer.

Outputs (saved to figures/C1/ or figures_last/C1/)
-------
  {model}_C1_gap.png
  {model}_C5_layer_drift_did.png
  {model}_C6_drift_did_overlay.png
  AGG_C1_gap.png

Usage
-----
  python analysis/02_refinement_distances.py
  python analysis/02_refinement_distances.py --pooling last
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, MODEL_COLORS, DATA_DIR, TABLES_DIR,
    ARM_REFINEMENT, ARM_COMMONALITY, ARM_LABELS, ARM_COLORS,
    load_activations, load_constraints_labels, load_metadata,
    normalised_depth_array, normalised_depth,
    get_cache_dir, get_figures_dir,
    save_figure,
    model_linestyle, setup_plt, SEED,
)
import json

setup_plt()
np.random.seed(SEED)

LEVELS   = [0, 1, 2, 3, 4]
N_CHAINS = 20
Z_95     = 1.96
RANK     = 128   # truncation rank for Mahalanobis SVD on constraints dataset

_COL_REF = ARM_COLORS[ARM_REFINEMENT]   # "#e15759"
_COL_COM = ARM_COLORS[ARM_COMMONALITY]  # "#4878d0"

K_DEFLATE    = 5    # top-K PCs removed before computing cosine
METRICS      = ["euc", "mah", "cos"]
METRIC_LABEL = {"euc": "Euclidean distance",
                "mah": "Mahalanobis d_M",
                "cos": f"Cosine (top-{K_DEFLATE} deflated)"}


# ── Level colour palette (viridis, 5 stops) ───────────────────────────────────

def _level_colors() -> list[str]:
    cmap  = mcm.get_cmap("viridis")
    stops = np.linspace(0.1, 0.9, 5)
    return [mcolors.to_hex(cmap(t)) for t in stops]

LEVEL_COLORS = _level_colors()


# ── Output directory ──────────────────────────────────────────────────────────

def _c1_dir(pooling: str, per_model: bool) -> Path:
    subfig = "refinement_geometry" if per_model else "refinement_geometry_agg"
    d = get_figures_dir(pooling, subfig)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, name: str, model: str = "", pooling: str = "mean") -> None:
    if model:
        out = _c1_dir(pooling, per_model=True) / f"{model}_{name}.png"
    else:
        out = _c1_dir(pooling, per_model=False) / f"{name}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    root = Path(__file__).resolve().parents[1]
    print(f"  saved → {out.relative_to(root)}", flush=True)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(model: str, pooling: str) -> Path:
    d = get_cache_dir("refinement", pooling)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}_{pooling}.npz"


def _save_cache(model: str, pooling: str, d: dict) -> None:
    np.savez(str(_cache_path(model, pooling)), **d)
    print(f"  [{model}] C1 cache saved.", flush=True)


def _load_cache(model: str, pooling: str) -> dict | None:
    p = _cache_path(model, pooling)
    if not p.exists():
        return None
    out = dict(np.load(str(p), allow_pickle=True))
    print(f"  [{model}] C1 loaded from cache.", flush=True)
    return out


# ── mean_no_last pooling (derived from mean + last + token counts) ───────────

def _n_tokens_path(model: str) -> Path:
    return get_cache_dir("refinement", "mean") / f"{model}_n_tokens_constraints.npy"


def _compute_n_tokens(model: str) -> np.ndarray:
    """
    Tokenize each constraints sentence with the model's tokenizer
    using the same call as inference (truncation=True, max_length=128).
    Returns an int32 array with one entry per sentence.
    """
    cache_p = _n_tokens_path(model)
    if cache_p.exists():
        n = np.load(str(cache_p))
        print(f"  [{model}] n_tokens loaded from cache.", flush=True)
        return n

    print(f"  [{model}] Tokenizing constraints sentences for n_tokens...",
          flush=True)
    from transformers import AutoTokenizer
    import os
    hf_id    = load_metadata(model)["hf_id"]

    # Pull HF_TOKEN from .env if not already in environment (for gated repos)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip()
                    break

    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentences: list[str] = []
    with open(DATA_DIR / "constraints" / "constraints_sentences.jsonl") as f:
        for line in f:
            sentences.append(json.loads(line)["sentence"])

    inputs = tokenizer(sentences, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
    n = inputs["attention_mask"].sum(dim=1).numpy().astype(np.int32)
    np.save(str(cache_p), n)
    print(f"  [{model}] n_tokens cached: min={n.min()}, max={n.max()}, "
          f"mean={n.mean():.2f}", flush=True)
    return n


def _load_pooled_activations(model: str, pooling: str) -> np.ndarray:
    """
    Returns a float32 (N, L, D) array.
    Supports 'mean', 'last', and 'mean_no_last' (derived algebraically).
    """
    if pooling in ("mean", "last"):
        return load_activations(model, "constraints", pooling)

    if pooling != "mean_no_last":
        raise ValueError(f"Unknown pooling: {pooling}")

    # Derive: mean_no_last = (N * mean - last) / (N - 1)
    print(f"  [{model}] Deriving mean_no_last pooling from mean + last...",
          flush=True)
    mean = np.asarray(load_activations(model, "constraints", "mean"),
                      dtype=np.float32)
    last = np.asarray(load_activations(model, "constraints", "last"),
                      dtype=np.float32)
    n_tok = _compute_n_tokens(model).astype(np.float32)        # (N,)

    # Guard: n_tok must be ≥ 2 to define mean_no_last
    if (n_tok < 2).any():
        bad = int((n_tok < 2).sum())
        print(f"  [{model}] WARNING: {bad} sentence(s) with n_tokens<2 — "
              f"setting their drift to NaN.", flush=True)

    n_b   = n_tok[:, None, None]                                # broadcast over (L, D)
    out   = (n_b * mean - last) / np.maximum(n_b - 1.0, 1.0)
    # Mark bad rows as NaN
    bad_mask = (n_tok < 2)
    if bad_mask.any():
        out[bad_mask] = np.nan
    return out


# ── Per-chain mean representations ───────────────────────────────────────────

def _per_chain_means(
    acts: np.ndarray,
    clab: dict,
    arm: str,
) -> np.ndarray:
    """
    Returns (n_chains, L, 5, D) per-chain mean representations.
    Chain order is alphabetical (stable across calls).
    """
    L = acts.shape[1]
    mask_a = clab["arm"] == arm
    chains  = np.unique(clab["chain_id"][mask_a])
    n       = len(chains)
    D       = acts.shape[2]
    out     = np.zeros((n, L, 5, D), dtype=np.float32)

    for ci, ch in enumerate(chains):
        for li, lv in enumerate(LEVELS):
            mask = mask_a & (clab["chain_id"] == ch) & (clab["level"] == lv)
            if mask.sum() == 0:
                continue
            out[ci, :, li, :] = acts[mask].mean(axis=0)   # (L, D)

    return out


# ── Distance computation (Euclidean, Mahalanobis, Cosine) ────────────────────

def _deflate(x: np.ndarray, Vt_k: np.ndarray) -> np.ndarray:
    """Remove top-K PC projections from vector x. Vt_k: (K, D)."""
    return x - Vt_k.T @ (Vt_k @ x)


def _euc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _cos(a: np.ndarray, b: np.ndarray,
         Vt_k: np.ndarray | None = None) -> float:
    """Cosine similarity; if Vt_k given, deflates both vectors first."""
    if Vt_k is not None:
        a = _deflate(a, Vt_k)
        b = _deflate(b, Vt_k)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _mah(a: np.ndarray, b: np.ndarray,
         Vt: np.ndarray, inv_sqrt_eig: np.ndarray) -> float:
    diff    = (a - b).astype(np.float64)
    diff_pc = diff @ Vt.T            # (R,)
    return float(np.linalg.norm(diff_pc * inv_sqrt_eig))


# ── Per-layer SVD for Mahalanobis (full constraints dataset) ─────────────────

def _layer_svd(
    X_layer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Thin randomized SVD of centered data at one layer.
    Returns:
      Vt           (RANK, D)   — all right singular vectors (for Mahalanobis)
      inv_sqrt_eig (RANK,)     — 1/sqrt(eigenvalue) per component
      Vt_k         (K_DEFLATE, D) — top-K rows reused for cosine deflation
    """
    X64 = X_layer.astype(np.float64)
    Xc  = X64 - X64.mean(axis=0)
    N, D = Xc.shape
    k    = min(RANK, N - 1, D - 1)
    np.random.seed(SEED)
    _, s, Vt     = randomized_svd(Xc, n_components=k, random_state=SEED)
    eigvals      = (s ** 2) / max(N - 1, 1)
    inv_sqrt_eig = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))
    Vt_k         = Vt[:K_DEFLATE].astype(np.float64)   # free — already computed
    return Vt.astype(np.float64), inv_sqrt_eig, Vt_k


# ── Main computation ──────────────────────────────────────────────────────────

def run(model: str, pooling: str = "mean", force: bool = False) -> dict | None:
    print(f"\n{'='*60}\nC1 — {model}  [{pooling}]\n{'='*60}", flush=True)

    cached = None if force else _load_cache(model, pooling)
    if cached is not None:
        return {"model": model, "pooling": pooling, **cached}

    acts    = _load_pooled_activations(model, pooling)
    N, L, D = acts.shape
    nd      = normalised_depth_array(L)
    clab    = load_constraints_labels()

    print(f"  [{model}] Computing per-chain means...", flush=True)
    ref_means = _per_chain_means(acts, clab, ARM_REFINEMENT)   # (20, L, 5, D)
    com_means = _per_chain_means(acts, clab, ARM_COMMONALITY)  # (20, L, 5, D)

    # ── Arrays to fill ────────────────────────────────────────────────────────
    # gap: for each reference level (5) × target level (5) × layer × metric
    # We store mean and CI for each arm separately.
    gap_ref  = {m: np.zeros((5, 5, L)) for m in METRICS}   # [ref_l, tgt_l, layer]
    gap_com  = {m: np.zeros((5, 5, L)) for m in METRICS}
    gap_ref_ci = {m: np.zeros((5, 5, L)) for m in METRICS}
    gap_com_ci = {m: np.zeros((5, 5, L)) for m in METRICS}

    # layer drift: distance from same-level mean at layer 0 → layer l
    # shape (5, L): [level, layer]
    drift_ref    = {m: np.zeros((5, L)) for m in METRICS}
    drift_com    = {m: np.zeros((5, L)) for m in METRICS}
    drift_ref_ci = {m: np.zeros((5, L)) for m in METRICS}
    drift_com_ci = {m: np.zeros((5, L)) for m in METRICS}

    print(f"  [{model}] Computing distances across {L} layers...", flush=True)

    for l in range(L):
        if l % 8 == 0:
            print(f"    layer {l}/{L}...", flush=True)

        Vt, inv_sqrt_eig, Vt_k = _layer_svd(acts[:, l, :])

        # Per-chain distance at this layer ─────────────────────────────────────
        # gap_ref: shape (n_chains, 5, 5) — [chain, ref_level, tgt_level]
        d_ref = {m: np.zeros((N_CHAINS, 5, 5)) for m in METRICS}
        d_com = {m: np.zeros((N_CHAINS, 5, 5)) for m in METRICS}

        for ci in range(N_CHAINS):
            for ri in range(5):
                for ti in range(5):
                    a_ref = ref_means[ci, l, ri, :].astype(np.float64)
                    b_ref = ref_means[ci, l, ti, :].astype(np.float64)
                    a_com = com_means[ci, l, ri, :].astype(np.float64)
                    b_com = com_means[ci, l, ti, :].astype(np.float64)

                    d_ref["euc"][ci, ri, ti] = _euc(a_ref, b_ref)
                    d_ref["cos"][ci, ri, ti] = _cos(a_ref, b_ref, Vt_k)
                    d_ref["mah"][ci, ri, ti] = _mah(a_ref, b_ref, Vt, inv_sqrt_eig)
                    d_com["euc"][ci, ri, ti] = _euc(a_com, b_com)
                    d_com["cos"][ci, ri, ti] = _cos(a_com, b_com, Vt_k)
                    d_com["mah"][ci, ri, ti] = _mah(a_com, b_com, Vt, inv_sqrt_eig)

        # Aggregate over chains ────────────────────────────────────────────────
        for m in METRICS:
            gap_ref[m][:, :, l]    = d_ref[m].mean(axis=0)
            gap_com[m][:, :, l]    = d_com[m].mean(axis=0)
            gap_ref_ci[m][:, :, l] = Z_95 * d_ref[m].std(axis=0, ddof=1) / np.sqrt(N_CHAINS)
            gap_com_ci[m][:, :, l] = Z_95 * d_com[m].std(axis=0, ddof=1) / np.sqrt(N_CHAINS)

        # Layer drift: distance from same-level mean at layer 0 → layer l ─────
        # Per-chain: d(chain_mean[c, 0, k, :], chain_mean[c, l, k, :])
        drift_ref_chain = {m: np.zeros((N_CHAINS, 5)) for m in METRICS}
        drift_com_chain = {m: np.zeros((N_CHAINS, 5)) for m in METRICS}
        for ci in range(N_CHAINS):
            for k in range(5):
                a_ref = ref_means[ci, 0, k, :].astype(np.float64)
                b_ref = ref_means[ci, l, k, :].astype(np.float64)
                a_com = com_means[ci, 0, k, :].astype(np.float64)
                b_com = com_means[ci, l, k, :].astype(np.float64)

                drift_ref_chain["euc"][ci, k] = _euc(a_ref, b_ref)
                drift_ref_chain["cos"][ci, k] = _cos(a_ref, b_ref, Vt_k)
                drift_ref_chain["mah"][ci, k] = _mah(a_ref, b_ref, Vt, inv_sqrt_eig)
                drift_com_chain["euc"][ci, k] = _euc(a_com, b_com)
                drift_com_chain["cos"][ci, k] = _cos(a_com, b_com, Vt_k)
                drift_com_chain["mah"][ci, k] = _mah(a_com, b_com, Vt, inv_sqrt_eig)

        for m in METRICS:
            drift_ref[m][:, l]    = drift_ref_chain[m].mean(axis=0)
            drift_com[m][:, l]    = drift_com_chain[m].mean(axis=0)
            drift_ref_ci[m][:, l] = Z_95 * drift_ref_chain[m].std(axis=0, ddof=1) / np.sqrt(N_CHAINS)
            drift_com_ci[m][:, l] = Z_95 * drift_com_chain[m].std(axis=0, ddof=1) / np.sqrt(N_CHAINS)

    # ── Build cache dict ──────────────────────────────────────────────────────
    cache_dict: dict = {"nd": nd, "L": L}
    for m in METRICS:
        cache_dict[f"gap_ref_{m}"]    = gap_ref[m]
        cache_dict[f"gap_com_{m}"]    = gap_com[m]
        cache_dict[f"gap_ref_ci_{m}"] = gap_ref_ci[m]
        cache_dict[f"gap_com_ci_{m}"] = gap_com_ci[m]
        cache_dict[f"drift_ref_{m}"]    = drift_ref[m]
        cache_dict[f"drift_com_{m}"]    = drift_com[m]
        cache_dict[f"drift_ref_ci_{m}"] = drift_ref_ci[m]
        cache_dict[f"drift_com_ci_{m}"] = drift_com_ci[m]

    _save_cache(model, pooling, cache_dict)

    return {"model": model, "pooling": pooling,
            "nd": nd, "L": L,
            "gap_ref": gap_ref, "gap_com": gap_com,
            "gap_ref_ci": gap_ref_ci, "gap_com_ci": gap_com_ci,
            "drift_ref": drift_ref, "drift_com": drift_com,
            "drift_ref_ci": drift_ref_ci, "drift_com_ci": drift_com_ci}


def _load_result(d: dict) -> dict:
    """Reconstruct nested dicts from flat cache keys."""
    nd = d["nd"]
    L  = int(d["L"])
    gap_ref    = {m: d[f"gap_ref_{m}"]    for m in METRICS}
    gap_com    = {m: d[f"gap_com_{m}"]    for m in METRICS}
    gap_ref_ci = {m: d[f"gap_ref_ci_{m}"] for m in METRICS}
    gap_com_ci = {m: d[f"gap_com_ci_{m}"] for m in METRICS}
    drift_ref    = {m: d[f"drift_ref_{m}"]    for m in METRICS}
    drift_com    = {m: d[f"drift_com_{m}"]    for m in METRICS}
    drift_ref_ci = {m: d[f"drift_ref_ci_{m}"] for m in METRICS}
    drift_com_ci = {m: d[f"drift_com_ci_{m}"] for m in METRICS}
    return {"nd": nd, "L": L,
            "gap_ref": gap_ref, "gap_com": gap_com,
            "gap_ref_ci": gap_ref_ci, "gap_com_ci": gap_com_ci,
            "drift_ref": drift_ref, "drift_com": drift_com,
            "drift_ref_ci": drift_ref_ci, "drift_com_ci": drift_com_ci}


# ── Figure: C1_gap (5 × 3) ───────────────────────────────────────────────────

def _fig_gap(res: dict, model: str) -> None:
    nd         = res["nd"]
    gap_ref    = res["gap_ref"]
    gap_com    = res["gap_com"]
    gap_ref_ci = res["gap_ref_ci"]
    gap_com_ci = res["gap_com_ci"]

    fig, axes = plt.subplots(5, 3, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"{model}  —  Distance from each reference level to all other levels\n"
        "Solid = Refinement | Dashed = Control | Bands = 95% CI | "
        "Colour = target level",
        fontsize=10, fontweight="bold",
    )

    for row, ref_l in enumerate(LEVELS):
        for col, m in enumerate(METRICS):
            ax = axes[row, col]

            for tgt_l in LEVELS:
                if tgt_l == ref_l:
                    continue
                col_tgt = LEVEL_COLORS[tgt_l]
                label   = f"S{tgt_l}"

                # Refinement — solid
                mean_r = gap_ref[m][ref_l, tgt_l]
                ci_r   = gap_ref_ci[m][ref_l, tgt_l]
                ax.plot(nd, mean_r, color=col_tgt, linestyle="-",
                        linewidth=1.8, label=label)
                ax.fill_between(nd, mean_r - ci_r, mean_r + ci_r,
                                color=col_tgt, alpha=0.15)

                # Control — dashed
                mean_c = gap_com[m][ref_l, tgt_l]
                ci_c   = gap_com_ci[m][ref_l, tgt_l]
                ax.plot(nd, mean_c, color=col_tgt, linestyle="--",
                        linewidth=1.8)
                ax.fill_between(nd, mean_c - ci_c, mean_c + ci_c,
                                color=col_tgt, alpha=0.08)

            if col == 0:
                ax.set_ylabel(f"S{ref_l} ref\n{METRIC_LABEL[m]}", fontsize=8)
            else:
                ax.set_ylabel(METRIC_LABEL[m], fontsize=7)
            if row == 0:
                ax.set_title(METRIC_LABEL[m], fontsize=9)
            if row == 4:
                ax.set_xlabel("Normalised depth", fontsize=8)

    # Single legend (target level colours + arm style)
    level_handles = [
        plt.Line2D([0], [0], color=LEVEL_COLORS[k], linewidth=2,
                   label=f"→S{k}") for k in LEVELS
    ]
    style_handles = [
        plt.Line2D([0], [0], color="dimgray", linestyle="-",  linewidth=2,
                   label="Refinement"),
        plt.Line2D([0], [0], color="dimgray", linestyle="--", linewidth=2,
                   label="Control"),
        matplotlib.patches.Patch(facecolor="dimgray", alpha=0.18,
                                 label="95% CI"),
    ]
    fig.legend(handles=level_handles + style_handles,
               fontsize=8, ncol=8, loc="lower center",
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "C1_gap", model=model, pooling=res.get("pooling", "mean"))


# ── Figure: C5_layer_drift_did (5 × 3) — diff-in-differences ────────────────

def _fig_layer_drift_did(res: dict, model: str) -> None:
    """
    Diff-in-differences of layer drift:
        d_DiD(t, k) = drift_ref(t, k) - drift_com(t, k)

    Subtracting control's drift removes the shared "general layer-by-layer
    motion" component, isolating what is specific to refinement at each level.
    The y-axis range is much smaller than the raw drifts, making the contrast
    easier to read.

    CI propagated as sqrt(CI_ref² + CI_com²) (independent samples — different
    chains in each arm).

    5 rows = levels S0..S4, 3 cols = Euclidean | Mahalanobis | Cosine.
    """
    nd           = res["nd"]
    drift_ref    = res["drift_ref"]
    drift_com    = res["drift_com"]
    drift_ref_ci = res["drift_ref_ci"]
    drift_com_ci = res["drift_com_ci"]

    fig, axes = plt.subplots(5, 3, figsize=(14, 18), sharex=True)
    fig.suptitle(
        f"{model}  —  Layer drift: refinement − control  (diff-in-differences)\n"
        "Positive = refinement drifts further from layer-0 anchor than control",
        fontsize=10, fontweight="bold",
    )

    for row, lv in enumerate(LEVELS):
        for col, m in enumerate(METRICS):
            ax = axes[row, col]

            did    = drift_ref[m][lv] - drift_com[m][lv]
            did_ci = np.sqrt(drift_ref_ci[m][lv] ** 2
                             + drift_com_ci[m][lv] ** 2)

            ax.plot(nd, did, color="black", linewidth=2.0)
            ax.fill_between(nd, did - did_ci, did + did_ci,
                            color="gray", alpha=0.20)
            ax.axhline(0.0, color="red", linestyle=":", linewidth=1.0)

            if col == 0:
                ax.set_ylabel(f"S{lv}\nΔ {METRIC_LABEL[m]}", fontsize=8)
            else:
                ax.set_ylabel(f"Δ {METRIC_LABEL[m]}", fontsize=7)
            if row == 0:
                ax.set_title(f"Δ {METRIC_LABEL[m]}", fontsize=9)
            if row == 4:
                ax.set_xlabel("Normalised depth", fontsize=8)

    style_handles = [
        plt.Line2D([0], [0], color="black", linewidth=2,
                   label="ref − control"),
        plt.Line2D([0], [0], color="red", linestyle=":", linewidth=1.5,
                   label="zero (no DiD)"),
        matplotlib.patches.Patch(facecolor="gray", alpha=0.20,
                                 label="propagated 95% CI"),
    ]
    fig.legend(handles=style_handles, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "C5_layer_drift_did", model=model,
          pooling=res.get("pooling", "mean"))


# ── Figure: C6_drift_did_overlay (1 × 3) — DiD by metric, all levels overlaid

def _fig_drift_did_overlay(res: dict, model: str) -> None:
    """
    All 5 levels of the DiD curve overlaid in one panel per metric.
    Lets you compare which level shows the largest refinement-vs-control
    difference at each layer. CI bands omitted to avoid overlap (use C5 for
    per-level CIs).
    """
    nd           = res["nd"]
    drift_ref    = res["drift_ref"]
    drift_com    = res["drift_com"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{model}  —  Diff-in-differences (refinement − control), all levels overlaid",
        fontsize=10, fontweight="bold",
    )

    for col, m in enumerate(METRICS):
        ax = axes[col]
        for lv in LEVELS:
            did = drift_ref[m][lv] - drift_com[m][lv]
            ax.plot(nd, did, color=LEVEL_COLORS[lv], linewidth=2.0,
                    label=f"S{lv}")

        ax.axhline(0.0, color="red", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Normalised depth")
        ax.set_ylabel(f"Δ {METRIC_LABEL[m]}  (ref − control)")
        ax.set_title(METRIC_LABEL[m])
        ax.legend(fontsize=8, title="Level", ncol=5, loc="best")

    fig.tight_layout()
    _save(fig, "C6_drift_did_overlay", model=model,
          pooling=res.get("pooling", "mean"))


# ── Aggregate figures (all models, one line per model) ───────────────────────

def _fig_agg_gap(all_res: list, pooling: str = "mean") -> None:
    """Aggregate C1_gap: one panel per metric, S0→S4 distance only, both arms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("C1 gap S0→S4 — all models  (refinement vs control)",
                 fontsize=10)

    for col, m in enumerate(METRICS):
        ax = axes[col]
        for res in all_res:
            model = res["model"]
            nd    = res["nd"]
            ls    = model_linestyle(model)
            col_m = MODEL_COLORS[model]
            ax.plot(nd, res["gap_ref"][m][0, 4], color=col_m, linestyle=ls,
                    linewidth=1.8, label=f"{model} Ref")
            ax.plot(nd, res["gap_com"][m][0, 4], color=col_m, linestyle=ls,
                    linewidth=1.0, alpha=0.5, label=f"{model} Ctrl")
        ax.set_title(METRIC_LABEL[m], fontsize=9)
        ax.set_xlabel("Normalised depth")
        if col == 0:
            ax.set_ylabel("Distance S0→S4")
        ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    _save(fig, "AGG_C1_gap", pooling=pooling)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="C1: level-to-level distances in constraints dataset")
    parser.add_argument("--models",  nargs="+", default=MODELS)
    parser.add_argument("--pooling", default="mean",
                        choices=["mean", "last", "mean_no_last"],
                        help="'mean_no_last' = average over all tokens except "
                             "the last (derived from mean + last + token counts)")
    parser.add_argument("--rank",    type=int, default=RANK,
                        help="Truncation rank for Mahalanobis SVD (default 128)")
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()

    RANK = args.rank   # allow CLI override

    all_res = []
    for model in args.models:
        raw = run(model, args.pooling, force=args.force)
        if raw is None:
            continue
        res = _load_result(raw) if "gap_ref_euc" in raw else raw
        res["model"]   = model
        res["pooling"] = args.pooling
        _fig_gap(res, model)
        _fig_layer_drift_did(res, model)
        _fig_drift_did_overlay(res, model)
        all_res.append(res)

    if all_res:
        _fig_agg_gap(all_res, args.pooling)

    print("\nC1_constraint_distances complete.", flush=True)
