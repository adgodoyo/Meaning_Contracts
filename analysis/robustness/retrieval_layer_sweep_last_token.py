"""
retrieval_layer_sweep_last_token.py -- per-language Recall@1 at every layer
under last-token readout, compared with mean pooling; for every language both
the value at normalised depth 0.5 and the best-layer value are reported
(Appendix G: peak depths per model and the Arabic values at the peak).

Outputs
-------
  results/robustness/retrieval_layer_sweep_last_token.csv
  results/robustness/retrieval_readout_comparison.csv
  results/robustness/retrieval_arabic_focus.csv
  results/figures_robustness/retrieval_last_token_by_layer.png
  results/figures_robustness/retrieval_readout_comparison_arabic.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FC_TABLES, FINAL_CODEBASE, MODELS, OUT_CSV_DIR, OUT_FIG_DIR,
    load_kmin, read_csv_rows, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    load_activations, load_multilingual_labels, top_k_deflation,
    cosine_pairwise, layer_at_depth, normalised_depth,
)

LANGUAGES = ["en", "de", "es", "ar", "zh"]
LANG_COLORS = {"en": "#1f77b4", "de": "#ff7f0e", "es": "#2ca02c",
               "ar": "#d62728", "zh": "#9467bd"}


def recall_at_1(sim_matrix: np.ndarray) -> float:
    top1 = np.argmax(sim_matrix, axis=1)
    return int(np.sum(top1 == np.arange(sim_matrix.shape[0]))) / sim_matrix.shape[0]


def per_pair_recall(X: np.ndarray, labels: dict) -> dict[tuple[str, str], float]:
    lang_arr, fid_arr = labels["language"], labels["factor_id"]
    out: dict[tuple[str, str], float] = {}
    for src in LANGUAGES:
        src_mask = lang_arr == src
        for tgt in LANGUAGES:
            if tgt == src:
                continue
            tgt_mask = lang_arr == tgt
            common = np.intersect1d(fid_arr[src_mask], fid_arr[tgt_mask])
            if len(common) == 0:
                continue
            src_idx = np.array([np.where(src_mask & (fid_arr == f))[0][0] for f in common])
            tgt_idx = np.array([np.where(tgt_mask & (fid_arr == f))[0][0] for f in common])
            out[(src, tgt)] = recall_at_1(cosine_pairwise(X[src_idx], X[tgt_idx]))
    return out


def per_language_overall(pair_recall: dict) -> dict[str, float]:
    out = {}
    for lang in LANGUAGES:
        vals = [v for (s, t), v in pair_recall.items() if s == lang]
        vals += [v for (s, t), v in pair_recall.items() if t == lang]
        out[lang] = float(np.mean(vals))
    return out


def load_paper_baseline(model: str) -> float:
    for row in read_csv_rows(FC_TABLES / "N6_retrieval_improvement.csv"):
        if row["model"] == model:
            return float(row["raw_r1_at_nd05"])
    raise KeyError(model)


def sweep(model: str, pooling: str, labels: dict) -> list[dict]:
    """Per-layer, per-language overall Recall@1 (deflated at that pooling's k_min)."""
    acts = load_activations(model, "multilingual", pooling)
    _, L, _ = acts.shape
    k_min = load_kmin(model, pooling)
    print(f"  {model} [{pooling}]: {L} layers, k_min={k_min}")
    rows = []
    for layer in range(L):
        X_def = top_k_deflation(acts[:, layer, :].astype(np.float32), k_min).astype(np.float32)
        overall = per_language_overall(per_pair_recall(X_def, labels))
        nd = normalised_depth(layer, L)
        for lang in LANGUAGES:
            rows.append({
                "model": model, "pooling": pooling, "layer": layer,
                "n_layers": L, "nd": round(float(nd), 4), "language": lang,
                "overall_recall_at_1": round(overall[lang], 4),
            })
    return rows


def main() -> None:
    labels = load_multilingual_labels()

    # ---- 1. last-token sweep (new) + mean sweep (recomputed here so both
    #         readouts come from one identical code path) --------------------
    all_rows: list[dict] = []
    for model in MODELS:
        for pooling in ("mean", "last"):
            all_rows.extend(sweep(model, pooling, labels))

    last_rows = [r for r in all_rows if r["pooling"] == "last"]
    write_csv(OUT_CSV_DIR / "retrieval_layer_sweep_last_token.csv", last_rows)

    # ---- 2. readout comparison: ND=0.5 point vs best layer ----------------
    comp_rows = []
    for model in MODELS:
        for pooling in ("mean", "last"):
            sub = [r for r in all_rows if r["model"] == model and r["pooling"] == pooling]
            L = sub[0]["n_layers"]
            nd05_layer = layer_at_depth(0.5, L)
            for lang in LANGUAGES:
                lang_rows = sorted([r for r in sub if r["language"] == lang],
                                   key=lambda r: r["layer"])
                vals = np.array([r["overall_recall_at_1"] for r in lang_rows])
                at_nd05 = float(vals[nd05_layer])
                best_i = int(np.argmax(vals))
                first_half = vals[: L // 2]
                second_half = vals[L // 2:]
                comp_rows.append({
                    "model": model, "pooling": pooling, "language": lang,
                    "recall_at_nd05": round(at_nd05, 4),
                    "best_layer": best_i,
                    "best_nd": round(float(lang_rows[best_i]["nd"]), 4),
                    "best_recall": round(float(vals[best_i]), 4),
                    "mean_first_half": round(float(first_half.mean()), 4),
                    "mean_second_half": round(float(second_half.mean()), 4),
                })
    write_csv(OUT_CSV_DIR / "retrieval_readout_comparison.csv", comp_rows)

    # ---- 3. Arabic focus: is 'ar weak' a depth/readout artifact? ----------
    ar_rows = []
    for model in MODELS:
        for pooling in ("mean", "last"):
            sub = [r for r in comp_rows if r["model"] == model and r["pooling"] == pooling]
            ar = next(r for r in sub if r["language"] == "ar")
            others = [r for r in sub if r["language"] != "ar"]
            # gap to the mean of the other four languages, at each vantage point
            ar_rows.append({
                "model": model, "pooling": pooling,
                "ar_at_nd05": ar["recall_at_nd05"],
                "others_mean_at_nd05": round(float(np.mean([r["recall_at_nd05"] for r in others])), 4),
                "gap_at_nd05": round(ar["recall_at_nd05"] - float(np.mean([r["recall_at_nd05"] for r in others])), 4),
                "ar_best": ar["best_recall"], "ar_best_nd": ar["best_nd"],
                "others_mean_best": round(float(np.mean([r["best_recall"] for r in others])), 4),
                "gap_at_best": round(ar["best_recall"] - float(np.mean([r["best_recall"] for r in others])), 4),
                "ar_first_half": ar["mean_first_half"],
                "others_first_half": round(float(np.mean([r["mean_first_half"] for r in others])), 4),
                "ar_second_half": ar["mean_second_half"],
                "others_second_half": round(float(np.mean([r["mean_second_half"] for r in others])), 4),
            })
    write_csv(OUT_CSV_DIR / "retrieval_arabic_focus.csv", ar_rows)

    # ---- printed summaries ------------------------------------------------
    for pooling in ("mean", "last"):
        print(f"\n=== Per-language overall Recall@1 — {pooling} pooling ===")
        print(f"{'model':14s}{'view':>12s}" + "".join(f"{l:>8s}" for l in LANGUAGES))
        for model in MODELS:
            sub = [r for r in comp_rows if r["model"] == model and r["pooling"] == pooling]
            for view, key in (("@ND=0.5", "recall_at_nd05"), ("best layer", "best_recall")):
                vals = [next(r for r in sub if r["language"] == l)[key] for l in LANGUAGES]
                print(f"{model:14s}{view:>12s}" + "".join(f"{v:8.3f}" for v in vals))

    print("\n=== Arabic vs. mean of other four languages ===")
    print(f"{'model':14s}{'pool':>6s}{'ar@.5':>8s}{'oth@.5':>8s}{'gap':>8s}"
          f"{'ar_1st':>8s}{'oth_1st':>8s}{'ar_2nd':>8s}{'oth_2nd':>8s}")
    for r in ar_rows:
        print(f"{r['model']:14s}{r['pooling']:>6s}{r['ar_at_nd05']:8.3f}"
              f"{r['others_mean_at_nd05']:8.3f}{r['gap_at_nd05']:8.3f}"
              f"{r['ar_first_half']:8.3f}{r['others_first_half']:8.3f}"
              f"{r['ar_second_half']:8.3f}{r['others_second_half']:8.3f}")

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        for lang in LANGUAGES:
            rows_ml = sorted([r for r in last_rows if r["model"] == model and r["language"] == lang],
                             key=lambda r: r["layer"])
            ax.plot([r["layer"] for r in rows_ml], [r["overall_recall_at_1"] for r in rows_ml],
                    color=LANG_COLORS[lang], label=lang, linewidth=1.6)
        ax.axhline(load_paper_baseline(model), color="red", linestyle="--", linewidth=1.2,
                   label="paper raw baseline @ND=0.5")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Layer"); ax.set_ylabel("Recall@1"); ax.set_ylim(0, 1)
    axes.flat[0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Per-language Recall@1 by layer (LAST-TOKEN readout, deflated)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG_DIR / "retrieval_last_token_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Arabic: mean vs last, per model
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        for pooling, style in (("mean", "-"), ("last", "--")):
            rows_ar = sorted([r for r in all_rows if r["model"] == model
                              and r["language"] == "ar" and r["pooling"] == pooling],
                             key=lambda r: r["layer"])
            ax.plot([r["nd"] for r in rows_ar], [r["overall_recall_at_1"] for r in rows_ar],
                    linestyle=style, color="#d62728", linewidth=1.8, label=f"ar ({pooling})")
            rows_oth = [r for r in all_rows if r["model"] == model
                        and r["language"] != "ar" and r["pooling"] == pooling]
            nds = sorted({r["nd"] for r in rows_oth})
            means = [float(np.mean([r["overall_recall_at_1"] for r in rows_oth if r["nd"] == nd]))
                     for nd in nds]
            ax.plot(nds, means, linestyle=style, color="#7f7f7f", linewidth=1.4,
                    label=f"other 4 langs ({pooling})")
        ax.axvline(0.5, color="black", linestyle=":", linewidth=1.0, label="paper ND=0.5")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Normalised depth"); ax.set_ylabel("Recall@1"); ax.set_ylim(0, 1)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Arabic vs. other languages, both readouts", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG_DIR / "retrieval_readout_comparison_arabic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    main()
