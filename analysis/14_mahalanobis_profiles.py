"""
14_mahalanobis_profiles.py — Per-layer Euclidean and Mahalanobis category
profiles under raw, top-k_min-deflated and language-residualized conditions.

Writes the cache read by Figure 2 (analysis/paper_figures/fig_image2_paper.py):

  results/caches/multilingual/{pooling}/A2/{model}_{pooling}.npz
      L, D, dip_layer, dip_nd,
      dm_{cond}_{C1..C7}  per-layer mean truncated-Mahalanobis distance (r=128)
      de_{cond}_{C1..C7}  per-layer mean Euclidean distance
      cond in {raw, deflated, residualized}; categories use the internal
      numbering of analysis/utils.PAIR_CATEGORIES (internal C2 = paper C1).

The computation is that of the original A2_metric_comparison analysis; its
exploratory figures and tables are not part of the paper and are omitted.

Usage
-----
  python analysis/14_mahalanobis_profiles.py                 # all models, mean pooling
  python analysis/14_mahalanobis_profiles.py --models llama3.2-3b --force
  python analysis/14_mahalanobis_profiles.py --out-dir /tmp/a2   # write elsewhere
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, PAIR_CATEGORIES, SEED,
    get_cache_dir, load_activations, load_multilingual_labels,
    build_multilingual_pair_indices, pairs_to_array,
    top_k_deflation, language_residualization,
    covariance_f64, svd_truncated,
    normalised_depth_array, normalised_depth,
    load_kmin, category_distances,
)

np.random.seed(SEED)

CATS = list(PAIR_CATEGORIES.keys())
RANK = 128
CONDITIONS = ["raw", "deflated", "residualized"]


def cache_path(model: str, pooling: str, out_dir: Path | None = None) -> Path:
    d = out_dir if out_dir is not None else get_cache_dir("multilingual", pooling) / "A2"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}_{pooling}.npz"


def run(model: str, pooling: str, force: bool, out_dir: Path | None) -> None:
    p = cache_path(model, pooling, out_dir)
    if p.exists() and not force:
        print(f"  [{model}] A2 cache exists: {p}", flush=True)
        return

    k_min = load_kmin(model)
    acts = load_activations(model, "multilingual", pooling)
    N, L, D = acts.shape
    labels = load_multilingual_labels()
    pair_lists = build_multilingual_pair_indices(labels)
    pairs = {c: pairs_to_array(pair_lists[c]) for c in CATS}
    print(f"\n[{model}] {pooling}  L={L}  D={D}  k_min={k_min}", flush=True)

    dm = {cond: {c: np.empty(L) for c in CATS} for cond in CONDITIONS}
    de = {cond: {c: np.empty(L) for c in CATS} for cond in CONDITIONS}
    for l in range(L):
        if l % 10 == 0:
            print(f"  layer {l}/{L}...", flush=True)
        X_raw = acts[:, l, :]
        X_def = top_k_deflation(X_raw, k_min).astype(np.float32)
        X_res = language_residualization(X_raw, labels["language"]).astype(np.float32)

        for cond, X in zip(CONDITIONS, [X_raw, X_def, X_res]):
            m_e, _ = category_distances(X, pairs, mode="euclidean")
            for c in CATS:
                de[cond][c][l] = m_e[c]
            cov = covariance_f64(X.astype(np.float64))
            vecs, vals = svd_truncated(cov, rank=RANK, model=model,
                                       dataset=f"multilingual_{cond}",
                                       pooling=pooling, layer=l)
            m_m, _ = category_distances(X, pairs, mode="mahalanobis", vecs=vecs, vals=vals)
            for c in CATS:
                dm[cond][c][l] = m_m[c]

    dip_layer = int(np.argmin(dm["raw"]["C2"]))
    out = {"L": L, "D": D, "dip_layer": dip_layer, "dip_nd": normalised_depth(dip_layer, L)}
    for cond in CONDITIONS:
        for c in CATS:
            out[f"dm_{cond}_{c}"] = dm[cond][c]
            out[f"de_{cond}_{c}"] = de[cond][c]
    np.savez(p, **out)
    print(f"  [{model}] saved {p}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    ap.add_argument("--out-dir", type=Path, default=None, help="write caches here instead")
    args = ap.parse_args()
    for m in args.models:
        run(m, args.pooling, args.force, args.out_dir)


if __name__ == "__main__":
    main()
