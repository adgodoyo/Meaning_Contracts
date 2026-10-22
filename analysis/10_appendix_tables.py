"""
10_appendix_tables.py — Generate all appendix tables that are not produced by scripts 01-09.

Tables produced
---------------
  tab:lv_summary    → logvolume_summary_full.csv   (16 rows: 4 models × 2 pooling × 2 conditions)
  tab:pr_summary    → pr_summary.csv               (8 rows: 4 models × 2 pooling, span-4 raw)
  tab:facet_dm      → facet_dm.csv                 (8 rows: 4 models × 2 pooling)
  kmin update       → kmin_per_model.csv            (adds last-token column)

Note: N6_retrieval_improvement.csv is produced by 11_retrieval_analysis.py.

Data sources
------------
  Raw log-vol / ID  : results/caches/refinement/{pooling}/B2/{model}.npz
  Top-5 log-vol / ID: results/caches/refinement/{pooling}/ZY/{model}_{arm}_{pooling}_id_lv.npz
  PR                : results/caches/refinement/{pooling}/B1_pr_cache/{model}_{arm}_{branch}.npz
  Mahalanobis       : results/caches/multilingual/{pooling}/{model}_{pooling}[_R128|_R256].npz
  Stable band       : results/tables/stable_band.csv

Usage
-----
  python analysis/10_appendix_tables.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, TABLES_DIR, SEED,
    get_cache_dir,
    load_multilingual_labels, build_multilingual_pair_indices, pairs_to_array,
    normalised_depth_array, load_stable_band,
    svd_truncated, mahalanobis_row_pairs, covariance_f64,
)

np.random.seed(SEED)

Z95       = 1.96
POOLINGS  = ["mean", "last"]
N_CHAINS  = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    return dict(np.load(str(path), allow_pickle=True))


def _summarize(diff: np.ndarray, lo: np.ndarray | None = None,
               hi: np.ndarray | None = None) -> dict:
    """Mean, AUC (sum), positive_frac, strict_positive_frac for a layer-wise curve."""
    diff = np.asarray(diff, dtype=np.float64)
    out = {
        "mean_gap":      round(float(np.nanmean(diff)), 2),
        "auc":           round(float(np.nansum(diff)), 2),
        "positive_frac": round(float(np.nanmean(diff > 0)) * 100),
    }
    if lo is not None and hi is not None:
        out["strict_frac"] = round(float(np.nanmean(lo > 0)) * 100)
    return out


def _span4_avg(arr: np.ndarray) -> np.ndarray:
    """Mean over chains of S4-S0 difference. arr: (N_chains, L, 5) → (L,)"""
    return (arr[:, :, 4] - arr[:, :, 0]).mean(axis=0)


def _span4_ci(arr: np.ndarray) -> np.ndarray:
    """95% CI half-width for span-4 per-chain difference. → (L,)"""
    per_chain = arr[:, :, 4] - arr[:, :, 0]   # (N_chains, L)
    return Z95 * per_chain.std(axis=0, ddof=1) / np.sqrt(N_CHAINS)


# ── Table 1: tab:lv_summary ───────────────────────────────────────────────────

def build_lv_summary() -> list[dict]:
    """
    16 rows: model × pooling × condition (raw, top-5 deflated).
    gap = (com_lv_S4 - com_lv_S0) - (ref_lv_S4 - ref_lv_S0)  per layer
    Positive = refinement more contractive (control has larger final neighborhood).
    """
    rows = []
    for pooling in POOLINGS:
        b2_dir  = get_cache_dir("refinement", pooling) / "B2"
        zy_dir  = get_cache_dir("refinement", pooling) / "ZY"

        for model in MODELS:
            # ── Raw log-volume (B2 cache) ─────────────────────────────────────
            b2 = _load(b2_dir / f"{model}.npz")
            ref_chain = b2["ref_chain_lv"]   # (N_chains, L, 5)
            com_chain = b2["com_chain_lv"]

            ref_mean = _span4_avg(ref_chain)    # (L,)
            ref_ci   = _span4_ci(ref_chain)
            com_mean = _span4_avg(com_chain)
            com_ci   = _span4_ci(com_chain)

            diff    = com_mean - ref_mean
            diff_lo = (com_mean - com_ci) - (ref_mean + ref_ci)
            diff_hi = (com_mean + com_ci) - (ref_mean - ref_ci)
            s = _summarize(diff, diff_lo, diff_hi)
            rows.append({"model": model, "pooling": pooling,
                          "condition": "raw", **s})

            # ── Top-5 deflated log-volume (ZY cache) ─────────────────────────
            ref_zy = _load(zy_dir / f"{model}_refinement_{pooling}_id_lv.npz")
            com_zy = _load(zy_dir / f"{model}_commonality_{pooling}_id_lv.npz")
            ref_chain5 = ref_zy["chain_lv"]    # (N_chains, L, 5)
            com_chain5 = com_zy["chain_lv"]

            ref_mean5 = _span4_avg(ref_chain5)
            ref_ci5   = _span4_ci(ref_chain5)
            com_mean5 = _span4_avg(com_chain5)
            com_ci5   = _span4_ci(com_chain5)

            diff5    = com_mean5 - ref_mean5
            diff5_lo = (com_mean5 - com_ci5) - (ref_mean5 + ref_ci5)
            diff5_hi = (com_mean5 + com_ci5) - (ref_mean5 - ref_ci5)
            s5 = _summarize(diff5, diff5_lo, diff5_hi)
            rows.append({"model": model, "pooling": pooling,
                          "condition": "top5", **s5})

    return rows


# ── Table 2: tab:pr_summary ───────────────────────────────────────────────────

def build_pr_summary() -> list[dict]:
    """
    8 rows: model × pooling. Raw branch, span-4 (S4-S0).
    gap = (ref_pr_S4 - ref_pr_S0) - (com_pr_S4 - com_pr_S0)  per layer.
    Negative = refinement has lower PR (more concentrated).
    """
    rows = []
    for pooling in POOLINGS:
        b1_dir = get_cache_dir("refinement", pooling) / "B1_pr_cache"

        for model in MODELS:
            ref = _load(b1_dir / f"{model}_refinement_raw.npz")["chain_pr"]
            com = _load(b1_dir / f"{model}_commonality_raw.npz")["chain_pr"]

            ref_span = ref[:, :, 4] - ref[:, :, 0]   # (N_chains, L)
            com_span = com[:, :, 4] - com[:, :, 0]

            diff    = ref_span.mean(axis=0) - com_span.mean(axis=0)
            ref_sd  = ref_span.std(axis=0, ddof=1)
            com_sd  = com_span.std(axis=0, ddof=1)
            ci_ref  = Z95 * ref_sd / np.sqrt(N_CHAINS)
            ci_com  = Z95 * com_sd / np.sqrt(N_CHAINS)
            diff_lo = (ref_span.mean(axis=0) - ci_ref) - (com_span.mean(axis=0) + ci_com)
            diff_hi = (ref_span.mean(axis=0) + ci_ref) - (com_span.mean(axis=0) - ci_com)

            # For PR summary: report positive AND negative strict fractions
            neg_lo  = -diff_hi
            neg_hi  = -diff_lo
            diff_f  = np.asarray(diff, dtype=np.float64)
            s = {
                "mean_gap":         round(float(np.nanmean(diff_f)), 2),
                "auc":              round(float(np.nansum(diff_f)), 2),
                "positive_frac":    round(float(np.nanmean(diff_f > 0)) * 100),
                "negative_frac":    round(float(np.nanmean(diff_f < 0)) * 100),
                "strict_pos_frac":  round(float(np.nanmean(diff_lo > 0)) * 100),
                "strict_neg_frac":  round(float(np.nanmean(diff_hi < 0)) * 100),
            }
            rows.append({"model": model, "pooling": pooling, **s})

    return rows


# ── Table 3: tab:facet_dm ─────────────────────────────────────────────────────

def _build_facet_pairs(labels: dict) -> tuple[list, list]:
    """
    Returns (same_facet_pairs, cross_facet_pairs) as lists of (i, j).

    Same-facet : C3 (same factor, diff language, independent) where sentence
                 position-index within the factor matches across languages.
    Cross-facet: C3 pairs where the position-index differs.

    Position-index is the 0-based index of the sentence within its factor × language
    group (sorted by occurrence in the dataset).
    """
    fid   = labels["factor_id"]
    lang  = labels["language"]
    stype = labels["sentence_type"]
    N     = len(fid)

    # Build position index: for each (factor, language) group, assign 0, 1, 2, ...
    pos_idx = np.full(N, -1, dtype=int)
    from collections import defaultdict
    group_counter: dict = defaultdict(int)
    for i in range(N):
        if stype[i] == "independent":
            key = (fid[i], lang[i])
            pos_idx[i] = group_counter[key]
            group_counter[key] += 1

    ind_idx = np.where(stype == "independent")[0]

    same_facet, cross_facet = [], []
    for ii, pi in enumerate(ind_idx):
        for pj in ind_idx[ii + 1:]:
            if fid[pi] == fid[pj] and lang[pi] != lang[pj]:
                if pos_idx[pi] == pos_idx[pj]:
                    same_facet.append((pi, pj))
                else:
                    cross_facet.append((pi, pj))

    return same_facet, cross_facet


def _mean_mahal(X: np.ndarray, pairs: list, stable_lo: int, stable_hi: int,
                pooling: str, model: str) -> float:
    """Mean Mahalanobis distance across layers in the stable band."""
    if not pairs:
        return float("nan")
    p_arr = np.array(pairs, dtype=int)

    lo, hi = stable_lo, min(stable_hi + 1, X.shape[1])
    dists = []
    R = 128 if pooling == "mean" else 256
    for l in range(lo, hi):
        Xl  = X[:, l, :].astype(np.float64)
        cov = covariance_f64(Xl)
        vecs, vals = svd_truncated(cov, rank=R)
        d = mahalanobis_row_pairs(Xl, p_arr, vecs, vals)
        dists.append(float(d.mean()))
    return float(np.mean(dists))


def build_facet_dm() -> list[dict]:
    """
    8 rows: model × pooling.
    Columns: exact_match_dm (C2), same_facet_dm (C3 same-index),
             cross_facet_dm (C3 diff-index), ratio (cross/same).
    """
    from utils import load_activations

    labels = load_multilingual_labels()
    c2_pairs = [p for p in build_multilingual_pair_indices(labels)["C2"]]
    same_pairs, cross_pairs = _build_facet_pairs(labels)
    print(f"  Facet pairs: same={len(same_pairs)}, cross={len(cross_pairs)}, C2={len(c2_pairs)}")

    rows = []
    for pooling in POOLINGS:
        R = 128 if pooling == "mean" else 256
        z2_dir = get_cache_dir("multilingual", pooling)

        for model in MODELS:
            print(f"  [{model}/{pooling}] loading activations...", flush=True)
            acts = load_activations(model, "multilingual", pooling)

            lo, hi = load_stable_band(model)

            # Use the Z2 multilingual cache for covariance (R=128 or R=256)
            # Compute directly from activations in the stable band
            c2_arr    = np.array(c2_pairs, dtype=int)
            same_arr  = np.array(same_pairs, dtype=int)
            cross_arr = np.array(cross_pairs, dtype=int)

            band_dists = {"c2": [], "same": [], "cross": []}
            for l in range(lo, min(hi + 1, acts.shape[1])):
                Xl  = acts[:, l, :].astype(np.float64)
                cov = covariance_f64(Xl)
                vecs, vals = svd_truncated(cov, rank=R)
                band_dists["c2"].append(
                    float(mahalanobis_row_pairs(Xl, c2_arr, vecs, vals).mean()))
                band_dists["same"].append(
                    float(mahalanobis_row_pairs(Xl, same_arr, vecs, vals).mean()))
                band_dists["cross"].append(
                    float(mahalanobis_row_pairs(Xl, cross_arr, vecs, vals).mean()))

            c2_dm    = round(float(np.mean(band_dists["c2"])), 2)
            same_dm  = round(float(np.mean(band_dists["same"])), 2)
            cross_dm = round(float(np.mean(band_dists["cross"])), 2)
            ratio    = round(cross_dm / same_dm, 2) if same_dm > 0 else float("nan")

            print(f"    [{model}/{pooling}] C2={c2_dm}, same-facet={same_dm}, "
                  f"cross-facet={cross_dm}, ratio={ratio}", flush=True)
            rows.append({
                "model": model, "pooling": pooling,
                "exact_dm": c2_dm, "same_facet_dm": same_dm,
                "cross_facet_dm": cross_dm, "ratio": ratio,
            })

    return rows


# ── kmin_per_model.csv update ─────────────────────────────────────────────────

def update_kmin_table() -> None:
    """Add last-token k_min column (paper-confirmed values) to kmin_per_model.csv."""
    # Paper-confirmed last-token kmin values
    kmin_last = {
        "pythia-2.8b": 10,
        "llama3.2-3b": 4,
        "qwen2.5-3b":  4,
        "gemma4-e4b":  4,
    }
    path = TABLES_DIR / "kmin_per_model.csv"
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))

    # Check if last column already present
    if rows and "k_min_last" in rows[0]:
        print("  kmin_per_model.csv: last-token column already present.", flush=True)
        return

    for row in rows:
        m = row["model"]
        row["k_min_last"] = kmin_last.get(m, "")

    fields = list(rows[0].keys()) if rows else ["model", "k_min", "k_min_last"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  kmin_per_model.csv updated with last-token k_min column.", flush=True)


# ── Write helpers ─────────────────────────────────────────────────────────────

def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path.name}  ({len(rows)} rows)", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate missing appendix tables")
    parser.add_argument("--skip-facet", action="store_true",
                        help="Skip facet_dm computation (needs activations)")
    args = parser.parse_args()

    print("\n--- tab:lv_summary ---")
    _write(TABLES_DIR / "logvolume_summary_full.csv", build_lv_summary())

    print("\n--- tab:pr_summary ---")
    _write(TABLES_DIR / "pr_summary.csv", build_pr_summary())

    print("\n--- kmin update ---")
    update_kmin_table()

    if not args.skip_facet:
        print("\n--- tab:facet_dm ---")
        _write(TABLES_DIR / "facet_dm.csv", build_facet_dm())
    else:
        print("\n--- tab:facet_dm skipped (--skip-facet) ---")

    print("\n10_appendix_tables complete.", flush=True)
