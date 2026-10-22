"""
retrieval_depth_sweep.py -- per-language-pair Recall@1 at normalised depths
0.35, 0.5 and 0.7, raw / deflated / residualised, all four models (Appendix G:
deflation improves retrieval for every language at every depth tested).

Output
------
  results/robustness/retrieval_depth_sweep.csv
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
ND_SWEEP = [0.35, 0.5, 0.7]
SAME_SCRIPT = {"en", "de", "es"}
CROSS_SCRIPT_LANGS = {"ar", "zh"}


def recall_at_1(sim_matrix: np.ndarray) -> float:
    Q = sim_matrix.shape[0]
    top1 = np.argmax(sim_matrix, axis=1)
    return int(np.sum(top1 == np.arange(Q))) / Q


def is_cross_script(src: str, tgt: str) -> bool:
    return (src in CROSS_SCRIPT_LANGS) != (tgt in CROSS_SCRIPT_LANGS)


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


def main() -> None:
    labels = load_multilingual_labels()
    rows = []
    for model in MODELS:
        acts = load_activations(model, "multilingual", "mean")
        N, L, D = acts.shape
        k_min = load_kmin(model, "mean")

        for nd in ND_SWEEP:
            layer = layer_at_depth(nd, L)
            X_raw = acts[:, layer, :].astype(np.float32)
            X_def = top_k_deflation(X_raw, k_min).astype(np.float32)
            X_res = language_residualization(X_raw, labels["language"]).astype(np.float32)

            for cond, X in [("raw", X_raw), ("deflated", X_def), ("residualized", X_res)]:
                pair_recall = per_pair_recall(X, labels)
                for (src, tgt), r1 in pair_recall.items():
                    rows.append({
                        "model": model, "nd": nd, "layer": layer, "condition": cond,
                        "src_lang": src, "tgt_lang": tgt, "recall_at_1": f"{r1:.4f}",
                        "cross_script": is_cross_script(src, tgt),
                    })

    write_csv(OUT_CSV_DIR / "retrieval_depth_sweep.csv", rows)

    print("\n--- Overall mean Recall@1 (all 20 pairs), by ND and condition ---")
    summary_lines = []
    for model in MODELS:
        for nd in ND_SWEEP:
            raw = np.mean([float(r["recall_at_1"]) for r in rows
                           if r["model"] == model and r["nd"] == nd and r["condition"] == "raw"])
            defl = np.mean([float(r["recall_at_1"]) for r in rows
                            if r["model"] == model and r["nd"] == nd and r["condition"] == "deflated"])
            within = np.mean([float(r["recall_at_1"]) for r in rows
                               if r["model"] == model and r["nd"] == nd and r["condition"] == "deflated"
                               and not r["cross_script"]])
            cross = np.mean([float(r["recall_at_1"]) for r in rows
                              if r["model"] == model and r["nd"] == nd and r["condition"] == "deflated"
                              and r["cross_script"]])
            line = (f"{model:14s} ND={nd:.2f} (layer={[r['layer'] for r in rows if r['model']==model and r['nd']==nd][0]}): "
                    f"raw={raw:.3f} deflated={defl:.3f} (+{100*(defl-raw):.1f}pts) | "
                    f"within-script={within:.3f} cross-script={cross:.3f} attenuation={within-cross:+.3f}")
            print(" ", line)
            summary_lines.append(line)



if __name__ == "__main__":
    main()
