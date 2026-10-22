"""
analysis/utils.py  —  Shared utilities for all analysis scripts.

Design principles
-----------------
* All statistical computations (covariance, distances, SVD) run in float64
  immediately after loading float32 activations.
* Expensive per-layer operations (covariance, truncated SVD) are disk-cached
  under results/caches/scratch/ so re-runs are fast.
* Pairwise distance computations on raw float32 tensors are GPU-accelerated
  via PyTorch (MPS on Apple Silicon, CUDA elsewhere) when the arrays are large
  enough to make the transfer overhead worthwhile.
* All random operations are seeded with SEED=42 for reproducibility.
* Pooling mode ("mean" or "last") is always passed explicitly — no env-var routing.

Cache layout
------------
  results/caches/multilingual/{pooling}/   — multilingual geometry caches
  results/caches/refinement/{pooling}/     — semantic-refinement caches
  results/caches/position/mean/            — position analysis (mean pooling only)
  results/caches/validation/retrieval/     — cross-lingual retrieval
  results/caches/scratch/covariances/      — intermediate covariance matrices
  results/caches/scratch/svd/              — intermediate SVD decompositions
  results/caches/scratch/residualized/     — language-residualized activations
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.stats import levene, wilcoxon
from sklearn.utils.extmath import randomized_svd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found — GPU acceleration disabled.", stacklevel=2)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent   # final_codebase root

ACTIVATIONS_DIR = _ROOT / "results" / "activations"
DATA_DIR        = _ROOT / "data"
TABLES_DIR      = _ROOT / "results" / "tables"
PAPER_FIGS_DIR  = _ROOT / "results" / "paper_figures"

# Scratch caches (intermediate computations: covariances, SVD, residualized)
_SCRATCH = _ROOT / "results" / "caches" / "scratch"


def get_cache_dir(dataset: str, pooling: str) -> Path:
    """
    Return the cache directory for a given dataset and pooling mode.

    dataset: "multilingual" | "refinement" | "position" | "validation/<sub>"
    pooling: "mean" | "last"
    """
    base = _ROOT / "results" / "caches"
    if dataset == "multilingual":
        return base / "multilingual" / pooling
    if dataset == "refinement":
        return base / "refinement" / pooling
    if dataset == "position":
        return base / "position" / "mean"  # position analysis is mean-only
    if dataset.startswith("validation/"):
        sub = dataset.split("/", 1)[1]
        return base / "validation" / sub
    # fallback
    return base / dataset / pooling


def get_figures_dir(pooling: str, subfig_type: str) -> Path:
    """
    Return the figures output directory.

    pooling:     "mean"  → results/figures_mean_pooling/{subfig_type}/
                 "last"  → results/figures_last_token/{subfig_type}/
    subfig_type: e.g. "multilingual_deflation", "refinement_geometry",
                      "position_analysis", "validation"
    """
    base = "figures_mean_pooling" if pooling == "mean" else "figures_last_token"
    return _ROOT / "results" / base / subfig_type


# Ensure scratch directories exist at import time
for _d in [
    TABLES_DIR, PAPER_FIGS_DIR,
    _SCRATCH / "covariances",
    _SCRATCH / "svd",
    _SCRATCH / "residualized",
]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

MODELS = ["pythia-2.8b", "llama3.2-3b", "qwen2.5-3b", "gemma4-e4b"]

# ---------------------------------------------------------------------------
# Constraint arm taxonomy
# ---------------------------------------------------------------------------

ARM_REFINEMENT    = "refinement"
ARM_COMMONALITY   = "commonality"
ARM_NEAR_SYNONYM  = "adv_near_synonym"
ARM_SCOPE_HEDGE   = "adv_scope_hedge"

ARMS_PRIMARY = [ARM_REFINEMENT, ARM_COMMONALITY]
ARMS_ADV     = [ARM_NEAR_SYNONYM, ARM_SCOPE_HEDGE]
ARMS_ALL     = ARMS_PRIMARY + ARMS_ADV

ARM_LABELS = {
    ARM_REFINEMENT:   "Refinement",
    ARM_COMMONALITY:  "Control",
    ARM_NEAR_SYNONYM: "Near-synonym",
    ARM_SCOPE_HEDGE:  "Scope hedge",
}

ARM_COLORS = {
    ARM_REFINEMENT:   "#e15759",
    ARM_COMMONALITY:  "#4878d0",
    ARM_NEAR_SYNONYM: "#59a14f",
    ARM_SCOPE_HEDGE:  "#8c564b",
}

# Semantic pair categories (multilingual probe) — ordered max → min similarity
PAIR_CATEGORIES = {
    "C1": "Same factor, same language",
    "C2": "Exact translation, diff language",
    "C3": "Same factor, diff language",
    "C4": "Same domain, diff factor, same lang",
    "C5": "Same domain, diff factor, diff lang",
    "C6": "Diff topic, same language",
    "C7": "Diff topic, diff language",
}

MODEL_COLORS = {
    "pythia-2.8b":  "#1f77b4",
    "llama3.2-3b":  "#ff7f0e",
    "qwen2.5-3b":   "#2ca02c",
    "gemma4-e4b":   "#d62728",
}

CATEGORY_COLORS = {
    "C1": "#1f77b4",
    "C2": "#ff7f0e",
    "C3": "#2ca02c",
    "C4": "#d62728",
    "C5": "#9467bd",
    "C6": "#8c564b",
    "C7": "#7f7f7f",
}

# ---------------------------------------------------------------------------
# GPU device selection
# ---------------------------------------------------------------------------

def _best_device() -> "torch.device":
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_GPU_DEVICE: Optional["torch.device"] = None


def get_device() -> "torch.device":
    global _GPU_DEVICE
    if _GPU_DEVICE is None and _TORCH_AVAILABLE:
        _GPU_DEVICE = _best_device()
    return _GPU_DEVICE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations(
    model: str,
    dataset: Literal["multilingual", "constraints"],
    pooling: Literal["mean", "last"] = "mean",
) -> np.ndarray:
    """
    Return a memory-mapped float32 array of shape (N, L, D).

    Cast to float64 before any numerical computation.
    """
    path = ACTIVATIONS_DIR / model / f"{model}_{dataset}_{pooling}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Activations not found: {path}")
    return np.load(str(path), mmap_mode="r")


def load_metadata(model: str) -> dict:
    path = ACTIVATIONS_DIR / model / f"{model}_metadata.json"
    with open(path) as f:
        return json.load(f)


def load_multilingual_labels() -> dict:
    """
    Return dict with keys: factor_id, domain, language, sentence_type.
    Each value is an array of length 1200 (40 factors × 5 languages × 6 sentences).
    """
    path = DATA_DIR / "multilingual" / "multilingual.jsonl"
    factor_ids, domains, languages, sentence_types = [], [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for s in rec["sentences"]:
                factor_ids.append(rec["factor_id"])
                domains.append(rec["domain"])
                languages.append(rec["language"])
                sentence_types.append(s["type"])
    return {
        "factor_id":     np.array(factor_ids),
        "domain":        np.array(domains),
        "language":      np.array(languages),
        "sentence_type": np.array(sentence_types),
    }


def load_constraints_labels() -> dict:
    """
    Return dict with keys: chain_id, arm, level, domain, surface_variant.
    Each value is an array of length 3840 (48 chains × 5 levels × 16 surface variants).
    """
    path = DATA_DIR / "constraints" / "constraints_sentences.jsonl"
    chain_ids, arms, levels, domains, variants = [], [], [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            chain_ids.append(rec["chain_id"])
            arms.append(rec["arm"])
            levels.append(int(rec["level"]))
            domains.append(rec.get("domain", ""))
            variants.append(int(rec.get("surface_variant", 0)))
    return {
        "chain_id":        np.array(chain_ids),
        "arm":             np.array(arms),
        "level":           np.array(levels),
        "domain":          np.array(domains),
        "surface_variant": np.array(variants),
    }


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def normalised_depth(layer_idx: int, n_layers: int) -> float:
    """ℓ / (L−1)  →  embedding layer = 0.0, last layer = 1.0."""
    if n_layers <= 1:
        return 0.0
    return layer_idx / (n_layers - 1)


def normalised_depth_array(n_layers: int) -> np.ndarray:
    return np.array([normalised_depth(i, n_layers) for i in range(n_layers)])


def layer_at_depth(target_depth: float, n_layers: int) -> int:
    depths = normalised_depth_array(n_layers)
    return int(np.argmin(np.abs(depths - target_depth)))


# ---------------------------------------------------------------------------
# GPU-accelerated pairwise distances (float32 in, float64 out)
# ---------------------------------------------------------------------------

def cosine_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (len(A), len(B)), float64."""
    use_gpu = _TORCH_AVAILABLE and max(len(A), len(B)) >= 256
    if use_gpu:
        dev = get_device()
        tA = torch.from_numpy(A.astype(np.float32)).to(dev)
        tB = torch.from_numpy(B.astype(np.float32)).to(dev)
        tA = tA / (tA.norm(dim=-1, keepdim=True) + 1e-12)
        tB = tB / (tB.norm(dim=-1, keepdim=True) + 1e-12)
        out = torch.mm(tA, tB.T)
        result = out.cpu().numpy().astype(np.float64)
        del tA, tB, out
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()
        return result
    A64 = A.astype(np.float64)
    B64 = B.astype(np.float64)
    An = A64 / (np.linalg.norm(A64, axis=-1, keepdims=True) + 1e-12)
    Bn = B64 / (np.linalg.norm(B64, axis=-1, keepdims=True) + 1e-12)
    return An @ Bn.T


def euclidean_pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix (len(A), len(B)), float64."""
    use_gpu = _TORCH_AVAILABLE and max(len(A), len(B)) >= 256
    if use_gpu:
        dev = get_device()
        tA = torch.from_numpy(A.astype(np.float32)).to(dev)
        tB = torch.from_numpy(B.astype(np.float32)).to(dev)
        A2 = (tA * tA).sum(dim=1, keepdim=True)
        B2 = (tB * tB).sum(dim=1, keepdim=True)
        dist2 = A2 + B2.T - 2.0 * torch.mm(tA, tB.T)
        dist2 = dist2.clamp(min=0.0)
        result = dist2.sqrt().cpu().numpy().astype(np.float64)
        del tA, tB, A2, B2, dist2
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()
        return result
    A64, B64 = A.astype(np.float64), B.astype(np.float64)
    A2 = np.sum(A64 ** 2, axis=1, keepdims=True)
    B2 = np.sum(B64 ** 2, axis=1, keepdims=True)
    dist2 = A2 + B2.T - 2.0 * (A64 @ B64.T)
    return np.sqrt(np.maximum(dist2, 0.0))


# ---------------------------------------------------------------------------
# Covariance (float64, cached in scratch/covariances/)
# ---------------------------------------------------------------------------

def _cov_cache_key(model: str, dataset: str, pooling: str, layer: int) -> Path:
    key = f"{model}_{dataset}_{pooling}_L{layer:03d}_cov.npy"
    return _SCRATCH / "covariances" / key


def covariance_f64(
    X: np.ndarray,
    model: str = "",
    dataset: str = "",
    pooling: str = "",
    layer: int = -1,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Compute (D, D) float64 covariance matrix of X (N, D).

    If model/dataset/pooling/layer are supplied, the result is cached to disk.
    """
    if use_cache and model and layer >= 0:
        cache_path = _cov_cache_key(model, dataset, pooling, layer)
        if cache_path.exists():
            return np.load(str(cache_path))

    X64 = X.astype(np.float64)
    mu  = X64.mean(axis=0)
    Xc  = X64 - mu
    cov = (Xc.T @ Xc) / (len(X64) - 1)

    if use_cache and model and layer >= 0:
        cache_path = _cov_cache_key(model, dataset, pooling, layer)
        np.save(str(cache_path), cov)

    return cov


# ---------------------------------------------------------------------------
# Truncated SVD of covariance (cached in scratch/svd/)
# ---------------------------------------------------------------------------

def _svd_cache_key(model: str, dataset: str, pooling: str, layer: int, rank: int) -> Path:
    key = f"{model}_{dataset}_{pooling}_L{layer:03d}_svd_r{rank:04d}.npz"
    return _SCRATCH / "svd" / key


def svd_truncated(
    cov: np.ndarray,
    rank: int = 128,
    model: str = "",
    dataset: str = "",
    pooling: str = "",
    layer: int = -1,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (vecs, vals):
        vecs  (D, rank)  — eigenvectors (columns)
        vals  (rank,)    — singular values (eigenvalues of covariance)
    """
    if use_cache and model and layer >= 0:
        cache_path = _svd_cache_key(model, dataset, pooling, layer, rank)
        if cache_path.exists():
            z = np.load(str(cache_path))
            return z["vecs"], z["vals"]

    np.random.seed(SEED)
    U, S, Vt = randomized_svd(cov, n_components=rank, random_state=SEED)
    vecs = Vt.T.astype(np.float64)   # (D, rank)
    vals = S.astype(np.float64)       # (rank,)

    if use_cache and model and layer >= 0:
        cache_path = _svd_cache_key(model, dataset, pooling, layer, rank)
        np.savez(str(cache_path), vecs=vecs, vals=vals)

    return vecs, vals


# ---------------------------------------------------------------------------
# Mahalanobis distance (truncated SVD version)
# ---------------------------------------------------------------------------

def mahalanobis_truncated(
    x: np.ndarray,
    xp: np.ndarray,
    vecs: np.ndarray,
    vals: np.ndarray,
    rank: Optional[int] = None,
) -> float:
    if rank is not None:
        vecs = vecs[:, :rank]
        vals = vals[:rank]
    diff = (x - xp).astype(np.float64)
    proj = vecs.T @ diff
    safe_vals = np.maximum(vals, 1e-10)
    d2 = np.sum(proj ** 2 / safe_vals)
    return float(np.sqrt(max(d2, 0.0)))


def mahalanobis_pairwise_truncated(
    A: np.ndarray,
    B: np.ndarray,
    vecs: np.ndarray,
    vals: np.ndarray,
    rank: Optional[int] = None,
) -> np.ndarray:
    """Pairwise truncated Mahalanobis distance matrix (len(A), len(B)), float64."""
    if rank is not None:
        vecs = vecs[:, :rank]
        vals = vals[:rank]
    safe_vals = np.maximum(vals, 1e-10)
    inv_sqrt  = 1.0 / np.sqrt(safe_vals)
    A64 = A.astype(np.float64) @ vecs * inv_sqrt
    B64 = B.astype(np.float64) @ vecs * inv_sqrt
    A2  = np.sum(A64 ** 2, axis=1, keepdims=True)
    B2  = np.sum(B64 ** 2, axis=1, keepdims=True)
    d2  = A2 + B2.T - 2.0 * (A64 @ B64.T)
    return np.sqrt(np.maximum(d2, 0.0))


# ---------------------------------------------------------------------------
# Spectral summary statistics
# ---------------------------------------------------------------------------

def participation_ratio(Sigma: np.ndarray) -> float:
    """PR(Σ) = (tr Σ)² / tr(Σ²) — effective dimensionality."""
    if Sigma.ndim == 1:
        lam = Sigma.astype(np.float64)
    else:
        lam = np.linalg.eigvalsh(Sigma.astype(np.float64))
    lam = np.maximum(lam, 0.0)
    tr  = float(np.sum(lam))
    tr2 = float(np.sum(lam ** 2))
    if tr2 < 1e-30:
        return 0.0
    return (tr ** 2) / tr2


def log_volume(Sigma: np.ndarray, rank: int = 128) -> float:
    """log det Σ approximated with the top-rank eigenvalues."""
    if Sigma.ndim == 1:
        lam = Sigma.astype(np.float64)
    else:
        from sklearn.utils.extmath import randomized_svd as _rsvd
        np.random.seed(SEED)
        _, lam, _ = _rsvd(
            Sigma.astype(np.float64),
            n_components=min(rank, Sigma.shape[0] - 1),
            random_state=SEED,
        )
    lam = lam[:rank]
    lam = np.maximum(lam, 1e-30)
    return float(np.sum(np.log(lam)))


# ---------------------------------------------------------------------------
# Nuisance removal
# ---------------------------------------------------------------------------

def top_k_deflation(X: np.ndarray, k: int) -> np.ndarray:
    """
    Remove the top-k principal components of X.

    X: (N, D). Returns mean-zero residuals with top-k PCs projected out.
    """
    X64 = X.astype(np.float64)
    mu  = X64.mean(axis=0)
    Xc  = X64 - mu

    np.random.seed(SEED)
    _, _, Vt = randomized_svd(Xc, n_components=k, random_state=SEED)

    if _TORCH_AVAILABLE and X.shape[0] * X.shape[1] >= 256 * 256:
        dev   = get_device()
        tXc   = torch.from_numpy(Xc.astype(np.float32)).to(dev)
        tVt   = torch.from_numpy(Vt.astype(np.float32)).to(dev)
        proj  = tXc @ tVt.T
        recon = proj @ tVt
        deflated = (tXc - recon).cpu().numpy().astype(np.float64)
        del tXc, tVt, proj, recon
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()
        return deflated

    proj  = Xc @ Vt.T
    recon = proj @ Vt
    return Xc - recon


def language_residualization(
    X: np.ndarray,
    lang_labels: np.ndarray,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Remove per-language conditional mean from X via OLS residualization.

    X: (N, D). Returns float64 array of same shape.
    """
    if cache_path is not None and cache_path.exists():
        return np.load(str(cache_path))

    X64 = X.astype(np.float64)
    langs = np.unique(lang_labels)
    residuals = X64.copy()
    for lang in langs:
        mask = lang_labels == lang
        residuals[mask] -= X64[mask].mean(axis=0)

    if cache_path is not None:
        np.save(str(cache_path), residuals)

    return residuals


def language_residualization_all_layers(
    acts: np.ndarray,
    lang_labels: np.ndarray,
    model: str,
    pooling: str = "mean",
    use_cache: bool = True,
) -> np.ndarray:
    """
    Apply language_residualization to every layer of acts (N, L, D).
    Returns float64 array (N, L, D). Cached in scratch/residualized/.
    """
    cache_file = _SCRATCH / "residualized" / f"{model}_multilingual_{pooling}_residualized.npy"
    if use_cache and cache_file.exists():
        return np.load(str(cache_file), mmap_mode="r")

    N, L, D = acts.shape
    out = np.empty((N, L, D), dtype=np.float64)
    for l in range(L):
        out[:, l, :] = language_residualization(acts[:, l, :], lang_labels)

    if use_cache:
        np.save(str(cache_file), out.astype(np.float32))
        return np.load(str(cache_file), mmap_mode="r")

    return out


# ---------------------------------------------------------------------------
# Intrinsic dimension
# ---------------------------------------------------------------------------

def mle_intrinsic_dim(X: np.ndarray, k: int = 5) -> float:
    """
    Levina–Bickel MLE intrinsic dimensionality estimator.

    X: (N, D). k: number of nearest neighbours.
    """
    X64 = X.astype(np.float64)
    D_mat = euclidean_pairwise(X64, X64)
    np.fill_diagonal(D_mat, np.inf)
    D_sorted = np.sort(D_mat, axis=1)[:, :k]
    eps = 1e-12
    D_sorted = np.maximum(D_sorted, eps)
    r_k   = D_sorted[:, -1:]
    r_j   = D_sorted[:, :-1]
    log_r = np.log(r_k / np.maximum(r_j, eps))
    m_i   = (k - 1) / np.sum(log_r, axis=1)
    return float(np.mean(m_i[np.isfinite(m_i)]))


# ---------------------------------------------------------------------------
# Innovation ratio
# ---------------------------------------------------------------------------

def innovation_ratio(H_l: np.ndarray, H_prev: np.ndarray, rank: int = 50) -> float:
    """
    Fraction of the layer-to-layer update that is orthogonal to the previous span.

    Returns scalar in [0, 1]: 0 = no new directions, 1 = fully novel directions.
    """
    H_l64  = H_l.astype(np.float64)
    H_p64  = H_prev.astype(np.float64)
    delta  = H_l64 - H_p64
    delta -= delta.mean(axis=0)
    H_pc   = H_p64 - H_p64.mean(axis=0)

    np.random.seed(SEED)
    _, _, Vt = randomized_svd(H_pc, n_components=min(rank, min(H_pc.shape) - 1),
                               random_state=SEED)
    proj  = (delta @ Vt.T) @ Vt
    ortho = delta - proj

    norm_ortho = float(np.linalg.norm(ortho, "fro"))
    norm_total = float(np.linalg.norm(delta, "fro"))

    if norm_total < 1e-30:
        return 0.0
    return norm_ortho / norm_total


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n: int = 2000,
    alpha: float = 0.05,
    stratify_by: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Return (lo, mid, hi) bootstrap confidence interval.

    stratify_by: if given, resample within groups (e.g. factor_id or chain_id).
    """
    np.random.seed(SEED)
    values = np.asarray(values, dtype=np.float64)
    mid = stat_fn(values)

    if stratify_by is not None:
        groups = np.unique(stratify_by)
        boot = np.empty(n)
        for i in range(n):
            idx = []
            for g in groups:
                g_idx = np.where(stratify_by == g)[0]
                idx.append(np.random.choice(g_idx, size=len(g_idx), replace=True))
            boot[i] = stat_fn(values[np.concatenate(idx)])
    else:
        boot = np.array([
            stat_fn(np.random.choice(values, size=len(values), replace=True))
            for _ in range(n)
        ])

    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, float(mid), hi


# ---------------------------------------------------------------------------
# Pair index generators for multilingual
# ---------------------------------------------------------------------------

def build_multilingual_pair_indices(labels: dict) -> dict[str, list[Tuple[int, int]]]:
    """
    For each of the 7 semantic pair categories, return a list of (i, j) index pairs.

    C1: Same factor, same language       — independent sentences
    C2: Exact translation, diff language — parallel sentences
    C3: Same factor, diff language       — independent sentences
    C4: Same domain, diff factor, same lang
    C5: Same domain, diff factor, diff lang
    C6: Diff topic, same language
    C7: Diff topic, diff language        — most distant

    C4–C7 are sampled (up to 10k pairs) to cap computation.
    """
    fid   = labels["factor_id"]
    dom   = labels["domain"]
    lang  = labels["language"]
    stype = labels["sentence_type"]
    N     = len(fid)
    idx   = np.arange(N)

    pairs: dict[str, list[Tuple[int, int]]] = {k: [] for k in PAIR_CATEGORIES}

    ind_mask = stype == "independent"
    ind_idx  = idx[ind_mask]
    for i, pi in enumerate(ind_idx):
        for pj in ind_idx[i + 1:]:
            if fid[pi] == fid[pj] and lang[pi] == lang[pj]:
                pairs["C1"].append((pi, pj))

    par_mask = stype == "parallel"
    par_idx  = idx[par_mask]
    for i, pi in enumerate(par_idx):
        for pj in par_idx[i + 1:]:
            if fid[pi] == fid[pj] and lang[pi] != lang[pj]:
                pairs["C2"].append((pi, pj))

    for i, pi in enumerate(ind_idx):
        for pj in ind_idx[i + 1:]:
            if fid[pi] == fid[pj] and lang[pi] != lang[pj]:
                pairs["C3"].append((pi, pj))

    rng = np.random.default_rng(SEED)
    MAX_CROSS = 10_000
    sample_size = min(N * (N - 1) // 2, 500_000)
    ii = rng.integers(0, N, size=sample_size)
    jj = rng.integers(0, N, size=sample_size)
    valid = ii < jj
    ii, jj = ii[valid], jj[valid]

    same_dom  = dom[ii]  == dom[jj]
    diff_fac  = fid[ii]  != fid[jj]
    same_lang = lang[ii] == lang[jj]
    diff_lang = ~same_lang
    diff_dom  = ~same_dom

    def _take(m: np.ndarray, cap: int) -> list[Tuple[int, int]]:
        idx_m = np.where(m)[0]
        if len(idx_m) > cap:
            idx_m = rng.choice(idx_m, size=cap, replace=False)
        return [(int(ii[k]), int(jj[k])) for k in idx_m]

    pairs["C4"] = _take(same_dom  & diff_fac & same_lang, MAX_CROSS)
    pairs["C5"] = _take(same_dom  & diff_fac & diff_lang, MAX_CROSS)
    pairs["C6"] = _take(diff_dom  & same_lang,            MAX_CROSS)
    pairs["C7"] = _take(diff_dom  & diff_lang,            MAX_CROSS)

    return pairs


# ---------------------------------------------------------------------------
# Stable-band helpers
# ---------------------------------------------------------------------------

STABLE_BAND_TABLE = TABLES_DIR / "stable_band.csv"


def save_stable_band(bands: dict[str, Tuple[int, int]], n_layers_map: dict[str, int]) -> None:
    """Write the stable band per model to stable_band.csv."""
    import csv
    with open(STABLE_BAND_TABLE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "l_lo", "l_hi", "nd_lo", "nd_hi", "n_layers"])
        for model, (lo, hi) in bands.items():
            L = n_layers_map[model]
            w.writerow([model, lo, hi,
                        f"{normalised_depth(lo, L):.4f}",
                        f"{normalised_depth(hi, L):.4f}",
                        L])
    print(f"  stable band saved → {STABLE_BAND_TABLE.relative_to(_ROOT)}")


def load_stable_band(model: str) -> Tuple[int, int]:
    """Read the stable band (l_lo, l_hi) for a given model from stable_band.csv."""
    import csv
    if not STABLE_BAND_TABLE.exists():
        raise FileNotFoundError(
            f"Stable band table not found at {STABLE_BAND_TABLE}. "
            "Run 01_multilingual_geometry.py first."
        )
    with open(STABLE_BAND_TABLE) as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                return int(row["l_lo"]), int(row["l_hi"])
    warnings.warn(f"Model '{model}' not in stable band table.")
    return 5, -5  # caller must add n_layers for the upper bound


def compute_stable_band(
    cosine_profiles: np.ndarray,
    min_width: int = 3,
) -> Tuple[int, int]:
    """
    Derive the stable band from cosine profile data.

    cosine_profiles: (L, C) — mean cosine per layer per category.
    Returns (l_lo, l_hi) — widest contiguous window where layer-to-layer
    spread variation is below the 25th percentile.
    """
    spread = cosine_profiles.max(axis=1) - cosine_profiles.min(axis=1)
    diff   = np.abs(np.diff(spread))
    threshold = np.percentile(diff, 25)
    stable = diff <= threshold

    best_lo, best_hi, best_len = 0, 0, 0
    lo = None
    for i, s in enumerate(stable):
        if s and lo is None:
            lo = i
        elif not s and lo is not None:
            length = i - lo
            if length > best_len:
                best_lo, best_hi, best_len = lo, i, length
            lo = None
    if lo is not None:
        length = len(stable) - lo
        if length > best_len:
            best_lo, best_hi, best_len = lo, len(stable), length

    if best_len < min_width:
        L = len(cosine_profiles)
        return L // 3, 2 * L // 3

    return best_lo, best_hi


# ---------------------------------------------------------------------------
# Two-stage peak-layer helpers
# ---------------------------------------------------------------------------

def gap_profile(
    metric_by_layer: np.ndarray,
    high_idx: int,
    low_idx: int,
) -> np.ndarray:
    """metric[high_idx] − metric[low_idx] across layers. Returns (L,) float64."""
    m = metric_by_layer.astype(np.float64)
    return m[:, high_idx] - m[:, low_idx]


def peak_layer(gap: np.ndarray, smooth: int = 3) -> int:
    gap = np.asarray(gap, dtype=np.float64)
    if smooth > 1 and len(gap) >= smooth:
        kernel   = np.ones(smooth) / smooth
        smoothed = np.convolve(gap, kernel, mode="same")
    else:
        smoothed = gap
    return int(np.argmax(smoothed))


def compute_metric_by_level_and_layer(
    acts: np.ndarray,
    labels: np.ndarray,
    level_values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    """
    For each (layer, level) compute metric_fn on the subset of activations.

    acts: (N, L, D). Returns (L, len(level_values)) float64.
    """
    L = acts.shape[1]
    K = len(level_values)
    result = np.empty((L, K), dtype=np.float64)
    for l in range(L):
        for ki, lv in enumerate(level_values):
            mask = labels == lv
            X = acts[mask, l, :].astype(np.float64)
            result[l, ki] = metric_fn(X)
    return result


# ---------------------------------------------------------------------------
# Mean-by-group helpers
# ---------------------------------------------------------------------------

def group_mean(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean of X (N, D) per unique label. Returns (unique_labels, means (K, D))."""
    unique = np.unique(labels)
    means  = np.stack([X[labels == u].mean(axis=0) for u in unique])
    return unique, means


# ---------------------------------------------------------------------------
# Row-wise distance helpers for indexed pairs (GPU-accelerated)
# ---------------------------------------------------------------------------

def cosine_row_pairs(acts: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for (P, 2) pairs. Returns (P,) float64."""
    A = acts[pairs[:, 0]].astype(np.float32)
    B = acts[pairs[:, 1]].astype(np.float32)

    if _TORCH_AVAILABLE and len(pairs) >= 256:
        dev = get_device()
        tA = torch.from_numpy(A).to(dev)
        tB = torch.from_numpy(B).to(dev)
        nA = tA.norm(dim=1, keepdim=True).clamp(min=1e-12)
        nB = tB.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cos = ((tA / nA) * (tB / nB)).sum(dim=1)
        result = cos.cpu().numpy().astype(np.float64)
        del tA, tB, nA, nB, cos
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()
        return result

    A64 = A.astype(np.float64)
    B64 = B.astype(np.float64)
    nA = np.linalg.norm(A64, axis=1) + 1e-12
    nB = np.linalg.norm(B64, axis=1) + 1e-12
    return np.sum((A64 / nA[:, None]) * (B64 / nB[:, None]), axis=1)


def euclidean_row_pairs(acts: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean distance for (P, 2) pairs. Returns (P,) float64."""
    A = acts[pairs[:, 0]].astype(np.float64)
    B = acts[pairs[:, 1]].astype(np.float64)
    return np.linalg.norm(A - B, axis=1)


def mahalanobis_row_pairs(
    acts: np.ndarray,
    pairs: np.ndarray,
    vecs: np.ndarray,
    vals: np.ndarray,
    rank: Optional[int] = None,
) -> np.ndarray:
    """Row-wise truncated Mahalanobis distance for (P, 2) pairs. Returns (P,) float64."""
    if rank is not None:
        vecs = vecs[:, :rank]
        vals = vals[:rank]

    inv_sqrt = 1.0 / np.sqrt(np.maximum(vals, 1e-10))
    diff = (acts[pairs[:, 0]].astype(np.float64)
            - acts[pairs[:, 1]].astype(np.float64))

    if _TORCH_AVAILABLE and len(pairs) >= 256:
        dev  = get_device()
        tD   = torch.from_numpy(diff.astype(np.float32)).to(dev)
        tV   = torch.from_numpy(vecs.astype(np.float32)).to(dev)
        tW   = torch.from_numpy(inv_sqrt.astype(np.float32)).to(dev)
        proj = (tD @ tV) * tW
        dists = proj.norm(dim=1).cpu().numpy().astype(np.float64)
        del tD, tV, tW, proj
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()
        return dists

    proj = diff @ vecs * inv_sqrt
    return np.linalg.norm(proj, axis=1)


# ---------------------------------------------------------------------------
# Per-category distance summary
# ---------------------------------------------------------------------------

def category_distances(
    acts_layer: np.ndarray,
    pairs_dict: dict,
    mode: str = "cosine",
    vecs: Optional[np.ndarray] = None,
    vals: Optional[np.ndarray] = None,
) -> Tuple[dict, dict]:
    """
    Mean and SD of pairwise distances per category at one layer.

    mode: "cosine" | "euclidean" | "mahalanobis"
    Returns (means, sds), each {cat: float}.
    """
    means, sds = {}, {}
    for cat, pairs in pairs_dict.items():
        if len(pairs) == 0:
            means[cat] = np.nan
            sds[cat]   = np.nan
            continue
        if mode == "cosine":
            d = cosine_row_pairs(acts_layer, pairs)
        elif mode == "euclidean":
            d = euclidean_row_pairs(acts_layer, pairs)
        elif mode == "mahalanobis":
            d = mahalanobis_row_pairs(acts_layer, pairs, vecs, vals)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        means[cat] = float(np.mean(d))
        sds[cat]   = float(np.std(d, ddof=1))
    return means, sds


# ---------------------------------------------------------------------------
# K_min table I/O  (written by 02_deflation_analysis.py)
# ---------------------------------------------------------------------------

KMIN_TABLE = TABLES_DIR / "kmin_per_model.csv"

# Deflation level used in the paper's figures, chosen from the Z2 deflation grids
# (Table 8 of the paper reports the full sweep): Pythia-2.8b k=8, Llama/Qwen/Gemma k=5
KMIN_PAPER = {
    "pythia-2.8b": 8,
    "llama3.2-3b": 5,
    "qwen2.5-3b":  5,
    "gemma4-e4b":  5,
}


def load_kmin(model: str, fallback: int = 5) -> int:
    """
    Read k_min for a model. Prefers kmin_per_model.csv; falls back to
    KMIN_PAPER (the deflation levels used in the paper's figures).
    """
    import csv
    if KMIN_TABLE.exists():
        with open(KMIN_TABLE) as f:
            for row in csv.DictReader(f):
                if row["model"] == model:
                    return int(row["k_min"])
    if model in KMIN_PAPER:
        return KMIN_PAPER[model]
    warnings.warn(f"Model '{model}' not in k_min table; using k={fallback}.", stacklevel=2)
    return fallback


# ---------------------------------------------------------------------------
# Matplotlib style and figure saving
# ---------------------------------------------------------------------------

def setup_plt() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi":        150,
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7,
        "lines.linewidth":   1.5,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save_figure(
    fig,
    pooling: str,
    subfig_type: str,
    name: str,
) -> Path:
    """
    Save a matplotlib figure.

    pooling:     "mean" or "last"
    subfig_type: subdirectory within figures_mean_pooling/ or figures_last_token/
    name:        filename without extension (e.g. "pythia-2.8b_cosine_profiles")
    """
    import matplotlib.pyplot as plt
    out_dir = get_figures_dir(pooling, subfig_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved → {out.relative_to(_ROOT)}")
    return out


def pairs_to_array(pairs_list: list) -> np.ndarray:
    """Convert list of (i, j) tuples to (P, 2) int array."""
    if not pairs_list:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs_list, dtype=np.int32)


def model_linestyle(model: str) -> str:
    return {
        "pythia-2.8b": "-",
        "llama3.2-3b": "--",
        "qwen2.5-3b":  "-.",
        "gemma4-e4b":  ":",
    }.get(model, "-")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("utils.py self-test")
    print(f"  ROOT         : {_ROOT}")
    print(f"  ACTIVATIONS  : {ACTIVATIONS_DIR}")
    print(f"  TABLES       : {TABLES_DIR}")
    print(f"  SCRATCH      : {_SCRATCH}")
    print(f"  TORCH        : {_TORCH_AVAILABLE}")
    if _TORCH_AVAILABLE:
        print(f"  DEVICE       : {get_device()}")

    print("\n--- cache dir helpers ---")
    for ds in ["multilingual", "refinement", "position", "validation/retrieval"]:
        for p in ["mean", "last"]:
            d = get_cache_dir(ds, p)
            print(f"  {ds:25s} {p}: {d.relative_to(_ROOT)}")

    print("\n--- figures dir helpers ---")
    for p in ["mean", "last"]:
        for sub in ["multilingual_deflation", "refinement_geometry"]:
            d = get_figures_dir(p, sub)
            print(f"  {p} {sub}: {d.relative_to(_ROOT)}")

    print("\n--- load_multilingual_labels ---")
    lab = load_multilingual_labels()
    print(f"  N={len(lab['factor_id'])}  "
          f"factors={len(np.unique(lab['factor_id']))}  "
          f"langs={np.unique(lab['language']).tolist()}")

    print("\n--- load_constraints_labels ---")
    clab = load_constraints_labels()
    print(f"  N={len(clab['chain_id'])}  "
          f"chains={len(np.unique(clab['chain_id']))}  "
          f"levels={np.unique(clab['level']).tolist()}  "
          f"arms={np.unique(clab['arm']).tolist()}")

    print("\n--- KMIN_PAPER ---")
    for m in MODELS:
        print(f"  {m}: k={load_kmin(m)}")

    print("\nSelf-test complete.")
