"""
retrieval_language_pairs.py -- cross-lingual Recall@1 for every directed
language pair (5 x 4), all four models, raw / top-k_min deflated /
language-residualised, mean pooling, normalised depth 0.5. Chance = 1/40.

Output
------
  results/robustness/retrieval_language_pairs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FINAL_CODEBASE, MODELS, OUT_CSV_DIR, load_kmin, write_csv

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    load_activations, load_multilingual_labels, top_k_deflation,
    language_residualization, cosine_pairwise, layer_at_depth,
)

LANGUAGES = ["en", "de", "es", "ar", "zh"]
TARGET_ND = 0.5


def recall_at_1(sim_matrix: np.ndarray) -> float:
    Q = sim_matrix.shape[0]
    top1 = np.argmax(sim_matrix, axis=1)
    hits = int(np.sum(top1 == np.arange(Q)))
    return hits / Q


def per_pair_recall(X: np.ndarray, labels: dict) -> dict[tuple[str, str], tuple[float, int]]:
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
            out[(src, tgt)] = (recall_at_1(sim), len(common))
    return out


def main() -> None:
    labels = load_multilingual_labels()
    rows = []
    for model in MODELS:
        acts = load_activations(model, "multilingual", "mean")
        N, L, D = acts.shape
        layer = layer_at_depth(TARGET_ND, L)
        k_min = load_kmin(model, "mean")

        X_raw = acts[:, layer, :].astype(np.float32)
        X_def = top_k_deflation(X_raw, k_min).astype(np.float32)
        X_res = language_residualization(X_raw, labels["language"]).astype(np.float32)

        for cond, X in [("raw", X_raw), ("deflated", X_def), ("residualized", X_res)]:
            pair_recall = per_pair_recall(X, labels)
            for (src, tgt), (r1, n) in pair_recall.items():
                rows.append({
                    "model": model, "layer": layer, "condition": cond,
                    "src_lang": src, "tgt_lang": tgt, "n_factors": n,
                    "recall_at_1": f"{r1:.4f}",
                })

    write_csv(OUT_CSV_DIR / "retrieval_language_pairs.csv", rows)

    # Cross-script vs within-script summary (ar and zh use non-Latin scripts).
    same_script = {"en", "de", "es"}  # Latin script
    cross_script_langs = {"ar", "zh"}

    def is_cross_script(src, tgt):
        return (src in cross_script_langs) != (tgt in cross_script_langs)

    print("\n--- Deflated Recall@1 by pair type, at ND=0.5 layer ---")
    summary_lines = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model and r["condition"] == "deflated"]
        within = [float(r["recall_at_1"]) for r in model_rows if not is_cross_script(r["src_lang"], r["tgt_lang"])]
        cross = [float(r["recall_at_1"]) for r in model_rows if is_cross_script(r["src_lang"], r["tgt_lang"])]
        line = (f"{model}: within-script mean R@1={np.mean(within):.3f} (n={len(within)} pairs), "
                f"cross-script mean R@1={np.mean(cross):.3f} (n={len(cross)} pairs), "
                f"attenuation={np.mean(within) - np.mean(cross):+.3f}")
        print(" ", line)
        summary_lines.append(line)

    # Weakest individual pair per model
    worst_lines = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model and r["condition"] == "deflated"]
        worst = min(model_rows, key=lambda r: float(r["recall_at_1"]))
        line = f"{model}: weakest pair {worst['src_lang']}->{worst['tgt_lang']} R@1={worst['recall_at_1']}"
        print(" ", line)
        worst_lines.append(line)



if __name__ == "__main__":
    main()
