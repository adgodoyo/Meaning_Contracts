"""
11_retrieval_analysis.py — Cross-Lingual Retrieval Improvement from Deflation
Converts geometry into a measurable downstream benefit.

Task: given a sentence in language A, retrieve its translation from language B
using nearest-neighbour search (cosine similarity).

Conditions: raw | top-K_min deflated | language-residualized
Metrics: Recall@1, Recall@5

Outputs
-------
results/figures/{pooling}/retrieval/{model}_N6_retrieval.png
results/figures/{pooling}/retrieval/AGG_N6_retrieval.png
results/tables/N6_retrieval_improvement.csv

Usage
-----
  python analysis/11_retrieval_analysis.py
  python analysis/11_retrieval_analysis.py --pooling last
  python analysis/11_retrieval_analysis.py --force
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS, MODEL_COLORS, TABLES_DIR, SEED,
    get_cache_dir,
    load_activations, load_multilingual_labels,
    top_k_deflation, language_residualization,
    cosine_pairwise,
    normalised_depth_array, normalised_depth, layer_at_depth,
    load_stable_band,
    model_linestyle, save_figure, setup_plt,
)

_KMIN_TABLE = TABLES_DIR / "kmin_per_model.csv"


def _load_kmin_n6(model: str, pooling: str) -> int:
    """Read k_min from kmin_per_model.csv (paper-confirmed values per pooling)."""
    col = "k_min" if pooling == "mean" else "k_min_last"
    if _KMIN_TABLE.exists():
        with open(_KMIN_TABLE) as f:
            for row in csv.DictReader(f):
                if row["model"] == model and col in row and row[col]:
                    return int(row[col])
    raise FileNotFoundError(
        f"kmin_per_model.csv not found or model '{model}' / column '{col}' missing."
    )

setup_plt()
np.random.seed(SEED)

LANGUAGES   = ["en", "de", "es", "ar", "zh"]
CONDITIONS  = ["raw", "deflated", "residualized"]
COND_LABELS = {
    "raw":          "Raw",
    "deflated":     "Top-K deflated",
    "residualized": "Language-residualized",
}


def _get_cache(pooling: str) -> Path:
    d = get_cache_dir("multilingual", pooling) / "N6"
    d.mkdir(parents=True, exist_ok=True)
    return d


def recall_at_k(sim_matrix: np.ndarray, k: int) -> float:
    """
    sim_matrix: (Q, D) — query × database cosine similarities.
    Assumes query[i] has its ground truth at database[i].
    Returns Recall@k.
    """
    Q     = sim_matrix.shape[0]
    top_k = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]
    hits  = sum(i in top_k[i] for i in range(Q))
    return hits / Q


def retrieval_per_layer(acts: np.ndarray, labels: dict, k_min: int) -> dict:
    """
    Compute Recall@1 and Recall@5 per layer per condition.
    Returns {cond: {"r1": (L,), "r5": (L,)}}
    """
    N, L, D  = acts.shape
    lang_arr = labels["language"]
    fid_arr  = labels["factor_id"]

    results = {c: {"r1": np.zeros(L), "r5": np.zeros(L)} for c in CONDITIONS}

    for l in range(L):
        if l % 5 == 0:
            print(f"    [retrieval] layer {l}/{L}", flush=True)
        X_raw = acts[:, l, :].astype(np.float32)
        X_def = top_k_deflation(X_raw, k_min).astype(np.float32)
        X_res = language_residualization(X_raw, lang_arr).astype(np.float32)

        for cond, X in zip(CONDITIONS, [X_raw, X_def, X_res]):
            r1_total = r5_total = n_pairs = 0

            for src_lang in LANGUAGES:
                src_mask = lang_arr == src_lang
                for tgt_lang in LANGUAGES:
                    if tgt_lang == src_lang:
                        continue
                    tgt_mask = lang_arr == tgt_lang

                    src_fids = fid_arr[src_mask]
                    tgt_fids = fid_arr[tgt_mask]
                    common   = np.intersect1d(src_fids, tgt_fids)
                    if len(common) == 0:
                        continue

                    src_idx = np.array([
                        np.where(src_mask & (fid_arr == f))[0][0]
                        for f in common
                    ])
                    tgt_idx = np.array([
                        np.where(tgt_mask & (fid_arr == f))[0][0]
                        for f in common
                    ])

                    sim = cosine_pairwise(X[src_idx], X[tgt_idx])
                    k1  = min(1, len(tgt_idx))
                    k5  = min(5, len(tgt_idx))
                    r1_total += recall_at_k(sim, k1) * len(common)
                    r5_total += recall_at_k(sim, k5) * len(common)
                    n_pairs  += len(common)

            if n_pairs > 0:
                results[cond]["r1"][l] = r1_total / n_pairs
                results[cond]["r5"][l] = r5_total / n_pairs

    return results


def run(model: str, pooling: str = "mean", force: bool = False) -> dict:
    print(f"\n{'='*60}\nN6 — {model}  [{pooling}]\n{'='*60}", flush=True)

    cache_p = _get_cache(pooling) / f"{model}.npz"

    if not force and cache_p.exists():
        c = dict(np.load(str(cache_p), allow_pickle=True))
        print(f"  [{model}] N6 loaded from cache", flush=True)
        L         = int(c["L"])
        nd        = normalised_depth_array(L)
        lo        = int(c["stable_lo"])
        hi        = int(c["stable_hi"])
        k_min     = int(c["k_min"])
        ret       = {cond: {"r1": c[f"r1_{cond}"], "r5": c[f"r5_{cond}"]}
                     for cond in CONDITIONS}
        peak_r1_l = int(c["peak_r1_layer"])
        baseline  = float(c["baseline"])
        improved  = float(c["improved"])
    else:
        acts    = load_activations(model, "multilingual", pooling)
        N, L, D = acts.shape
        nd      = normalised_depth_array(L)
        labels  = load_multilingual_labels()
        lo, hi  = load_stable_band(model)
        k_min   = _load_kmin_n6(model, pooling)
        print(f"  [{model}] K_min={k_min}  |  stable band {lo}–{hi}", flush=True)
        print(f"  [{model}] Computing retrieval across {L} layers...", flush=True)
        ret       = retrieval_per_layer(acts, labels, k_min)
        peak_r1_l = int(np.argmax(ret["deflated"]["r1"]))
        baseline  = float(ret["raw"]["r1"][peak_r1_l])
        improved  = float(ret["deflated"]["r1"][peak_r1_l])
        print(f"  [{model}] Raw R@1 at layer {peak_r1_l}: {baseline:.3f}", flush=True)
        print(f"  [{model}] Deflated R@1 at layer {peak_r1_l}: {improved:.3f}  "
              f"(+{100*(improved-baseline):.1f}%)", flush=True)

        cache_dict: dict = {
            "L": L, "stable_lo": lo, "stable_hi": hi,
            "k_min": k_min, "peak_r1_layer": peak_r1_l,
            "baseline": baseline, "improved": improved,
        }
        for cond in CONDITIONS:
            cache_dict[f"r1_{cond}"] = ret[cond]["r1"]
            cache_dict[f"r5_{cond}"] = ret[cond]["r5"]
        np.savez(str(cache_p), **cache_dict)
        print(f"  [{model}] N6 cache saved", flush=True)

    # ── Per-model figure ──────────────────────────────────────────────────────
    cond_colors = {"raw": "gray", "deflated": "steelblue", "residualized": "tomato"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle(f"{model}  [{pooling}]  —  Cross-lingual retrieval", fontsize=11)

    for ax, metric, title in [
        (axes[0], "r1", "Recall@1"),
        (axes[1], "r5", "Recall@5"),
    ]:
        for cond in CONDITIONS:
            ax.plot(nd, ret[cond][metric],
                    color=cond_colors[cond], linewidth=1.5,
                    label=COND_LABELS[cond])
        ax.axvspan(normalised_depth(lo, L), normalised_depth(hi, L),
                   alpha=0.10, color="green", zorder=0, label="Stable band")
        ax.set_xlabel("Normalised depth  ℓ/(L−1)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

    save_figure(fig, pooling, "retrieval", f"{model}_N6_retrieval")

    nd05_l = layer_at_depth(0.5, L)
    agg_r1 = {cond: float(ret[cond]["r1"][nd05_l]) for cond in CONDITIONS}

    return {
        "model": model, "L": L, "nd": nd,
        "ret": ret, "k_min": k_min,
        "agg_r1_at_nd05": agg_r1,
        "peak_r1_layer": peak_r1_l,
        "baseline": baseline, "improved": improved,
    }


def aggregate(all_results: list, pooling: str) -> None:
    print("\nN6 — Aggregate figure", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Cross-lingual retrieval improvement  [{pooling}]  — all models",
                 fontsize=11)

    ax = axes[0]
    for res in all_results:
        ax.plot(res["nd"], res["ret"]["deflated"]["r1"],
                color=MODEL_COLORS[res["model"]],
                linestyle=model_linestyle(res["model"]),
                label=res["model"], linewidth=1.5)
    ax.set_xlabel("Normalised depth")
    ax.set_ylabel("Recall@1 (deflated)")
    ax.set_title("Cross-lingual Recall@1 — top-K deflated")
    ax.legend()
    ax.set_ylim(0, 1)

    ax2    = axes[1]
    models = [r["model"] for r in all_results]
    x      = np.arange(len(models))
    width  = 0.25
    cond_colors = {"raw": "lightgray", "deflated": "steelblue", "residualized": "tomato"}
    for i, cond in enumerate(CONDITIONS):
        vals = [r["agg_r1_at_nd05"][cond] for r in all_results]
        ax2.bar(x + (i - 1) * width, vals, width,
                color=cond_colors[cond], label=COND_LABELS[cond],
                edgecolor="k", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Recall@1")
    ax2.set_title("Recall@1 at ND=0.5 (three conditions)")
    ax2.legend(fontsize=7)
    ax2.set_ylim(0, 1)

    save_figure(fig, pooling, "retrieval", "AGG_N6_retrieval")


def save_tables(all_results: list) -> None:
    path = TABLES_DIR / "N6_retrieval_improvement.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "k_min", "peak_layer",
                    "raw_r1_at_peak", "deflated_r1_at_peak",
                    "improvement_pct",
                    "raw_r1_at_nd05", "deflated_r1_at_nd05",
                    "resid_r1_at_nd05"])
        for res in all_results:
            w.writerow([
                res["model"], res["k_min"], res["peak_r1_layer"],
                f"{res['baseline']:.4f}",
                f"{res['improved']:.4f}",
                f"{100*(res['improved']-res['baseline']):.2f}",
                f"{res['agg_r1_at_nd05']['raw']:.4f}",
                f"{res['agg_r1_at_nd05']['deflated']:.4f}",
                f"{res['agg_r1_at_nd05']['residualized']:.4f}",
            ])
    print(f"  Retrieval table → {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="N6: Cross-lingual retrieval improvement from deflation")
    parser.add_argument("--models",  nargs="+", default=MODELS)
    parser.add_argument("--pooling", default="mean", choices=["mean", "last"])
    parser.add_argument("--force",   action="store_true",
                        help="Recompute even if cache exists")
    args = parser.parse_args()

    all_results = [run(m, args.pooling, force=args.force) for m in args.models]
    aggregate(all_results, args.pooling)
    save_tables(all_results)
    print("\nN6 complete.", flush=True)
