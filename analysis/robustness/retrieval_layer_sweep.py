"""
retrieval_layer_sweep.py -- per-language Recall@1 (as source, as target, overall)
at every layer, deflated condition, mean pooling, all four models; per-language
tables at normalised depths 0.35 and 0.75 (Section 4.1: Gemma reaches 0.700 at
depth 0.35).

Outputs
-------
  results/robustness/retrieval_layer_sweep.csv
  results/robustness/retrieval_layer_sweep_nd35.csv
  results/robustness/retrieval_layer_sweep_nd75.csv
  results/figures_robustness/retrieval_layer_sweep_{source,target,overall}_by_layer.png
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
LANG_COLORS = {"en": "#1f77b4", "de": "#ff7f0e", "es": "#2ca02c", "ar": "#d62728", "zh": "#9467bd"}
ND_TABLE_POINTS = [0.35, 0.75]


def recall_at_1(sim_matrix: np.ndarray) -> float:
    Q = sim_matrix.shape[0]
    top1 = np.argmax(sim_matrix, axis=1)
    return int(np.sum(top1 == np.arange(Q))) / Q


def per_pair_recall(X: np.ndarray, labels: dict) -> dict[tuple[str, str], float]:
    lang_arr, fid_arr = labels["language"], labels["factor_id"]
    out = {}
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
            sim = cosine_pairwise(X[src_idx], X[tgt_idx])
            out[(src, tgt)] = recall_at_1(sim)
    return out


def per_language_summaries(pair_recall: dict) -> dict[str, dict[str, float]]:
    """Returns {lang: {source, target, overall}}."""
    out = {}
    for lang in LANGUAGES:
        src_vals = [v for (s, t), v in pair_recall.items() if s == lang]
        tgt_vals = [v for (s, t), v in pair_recall.items() if t == lang]
        out[lang] = {
            "source": float(np.mean(src_vals)),
            "target": float(np.mean(tgt_vals)),
            "overall": float(np.mean(src_vals + tgt_vals)),
        }
    return out


def load_paper_baseline(model: str) -> float:
    for row in read_csv_rows(FC_TABLES / "N6_retrieval_improvement.csv"):
        if row["model"] == model:
            return float(row["raw_r1_at_nd05"])
    raise KeyError(model)


def main() -> None:
    labels = load_multilingual_labels()
    full_rows = []
    table_rows = {nd: [] for nd in ND_TABLE_POINTS}

    for model in MODELS:
        acts = load_activations(model, "multilingual", "mean")
        N, L, D = acts.shape
        k_min = load_kmin(model, "mean")
        print(f"{model}: sweeping {L} layers (k_min={k_min})...")

        for layer in range(L):
            X_raw = acts[:, layer, :].astype(np.float32)
            X_def = top_k_deflation(X_raw, k_min).astype(np.float32)
            pair_recall = per_pair_recall(X_def, labels)
            lang_summ = per_language_summaries(pair_recall)
            nd = normalised_depth(layer, L)

            for lang in LANGUAGES:
                full_rows.append({
                    "model": model, "layer": layer, "nd": f"{nd:.4f}", "language": lang,
                    "source_recall_at_1": f"{lang_summ[lang]['source']:.4f}",
                    "target_recall_at_1": f"{lang_summ[lang]['target']:.4f}",
                    "overall_recall_at_1": f"{lang_summ[lang]['overall']:.4f}",
                })

        # Table snapshots at requested ND points (nearest-layer, deflated)
        for nd_target in ND_TABLE_POINTS:
            layer = layer_at_depth(nd_target, L)
            X_raw = acts[:, layer, :].astype(np.float32)
            X_def = top_k_deflation(X_raw, k_min).astype(np.float32)
            pair_recall = per_pair_recall(X_def, labels)
            lang_summ = per_language_summaries(pair_recall)
            for lang in LANGUAGES:
                table_rows[nd_target].append({
                    "model": model, "requested_nd": nd_target, "actual_layer": layer,
                    "actual_nd": f"{normalised_depth(layer, L):.4f}", "language": lang,
                    "source_recall_at_1": f"{lang_summ[lang]['source']:.4f}",
                    "target_recall_at_1": f"{lang_summ[lang]['target']:.4f}",
                    "overall_recall_at_1": f"{lang_summ[lang]['overall']:.4f}",
                })

    write_csv(OUT_CSV_DIR / "retrieval_layer_sweep.csv", full_rows)
    for nd in ND_TABLE_POINTS:
        write_csv(OUT_CSV_DIR / f"retrieval_layer_sweep_nd{str(nd).replace('0.', '')}.csv", table_rows[nd])

    # --- print the two requested tables ---
    for nd in ND_TABLE_POINTS:
        for modality in ("source", "target", "overall"):
            print(f"\n=== ND={nd} — per-language {modality} Recall@1 (deflated) ===")
            header = "{:14s}".format("model") + "".join(f"{l:>8s}" for l in LANGUAGES)
            print(header)
            for model in MODELS:
                vals = [next(r for r in table_rows[nd] if r["model"] == model and r["language"] == l)[f"{modality}_recall_at_1"]
                        for l in LANGUAGES]
                print("{:14s}".format(model) + "".join(f"{float(v):8.3f}" for v in vals))

    # --- figures: one per modality, 2x2 subplots by model ---
    for modality in ("source", "target", "overall"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
        for ax, model in zip(axes.flat, MODELS):
            baseline = load_paper_baseline(model)
            for lang in LANGUAGES:
                xs = [int(r["layer"]) for r in full_rows if r["model"] == model and r["language"] == lang]
                ys = [float(r[f"{modality}_recall_at_1"]) for r in full_rows if r["model"] == model and r["language"] == lang]
                ax.plot(xs, ys, color=LANG_COLORS[lang], label=lang, linewidth=1.6)
            ax.axhline(baseline, color="red", linestyle="--", linewidth=1.2,
                       label=f"paper raw baseline @ND=0.5 ({baseline:.3f})")
            ax.set_title(model, fontsize=10)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Recall@1")
            ax.set_ylim(0, 1)
        axes.flat[0].legend(fontsize=7, loc="upper left", ncol=2)
        fig.suptitle(f"Per-language Recall@1 by layer ({modality}, deflated, top-k_min)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_png = OUT_FIG_DIR / f"retrieval_layer_sweep_{modality}_by_layer.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_png}")



if __name__ == "__main__":
    main()
