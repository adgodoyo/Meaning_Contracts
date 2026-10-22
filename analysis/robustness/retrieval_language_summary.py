"""
retrieval_language_summary.py -- per-language Recall@1 marginals (each language
as source and as target, and their average), the 5 x 5 pair matrix and
script-block summaries, from retrieval_language_pairs.csv (Table 10, Appendix G).

Outputs
-------
  results/robustness/retrieval_language_marginals.csv
  results/robustness/retrieval_language_pair_matrix.csv
  results/robustness/retrieval_script_blocks.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, OUT_CSV_DIR, read_csv_rows, write_csv

SRC = OUT_CSV_DIR / "retrieval_language_pairs.csv"
LANGS = ["en", "de", "es", "ar", "zh"]
EURO = {"en", "de", "es"}
CHANCE = 1.0 / 40


def block_of(a: str, b: str) -> str:
    ea, eb = a in EURO, b in EURO
    if ea and eb:
        return "Euro<->Euro"
    if ea != eb:
        other = b if ea else a
        return f"Euro<->{other}"
    return "Ar<->Zh"          # both non-European, and they are the only two


def main() -> None:
    rows = read_csv_rows(SRC)
    conditions = ["raw", "deflated"]

    # ---- (a) ordered-pair matrix ----
    matrix_rows: list[dict] = []
    for cond in conditions:
        for model in MODELS + ["MEAN_OF_4_MODELS"]:
            for s in LANGS:
                for t in LANGS:
                    if s == t:
                        continue
                    if model == "MEAN_OF_4_MODELS":
                        vals = [float(r["recall_at_1"]) for r in rows
                                if r["condition"] == cond and r["src_lang"] == s
                                and r["tgt_lang"] == t]
                    else:
                        vals = [float(r["recall_at_1"]) for r in rows
                                if r["condition"] == cond and r["model"] == model
                                and r["src_lang"] == s and r["tgt_lang"] == t]
                    if not vals:
                        continue
                    matrix_rows.append({
                        "condition": cond, "model": model,
                        "src_lang": s, "tgt_lang": t,
                        "block": block_of(s, t),
                        "recall_at_1": f"{np.mean(vals):.4f}",
                        "times_chance": f"{np.mean(vals) / CHANCE:.1f}",
                    })
    write_csv(OUT_CSV_DIR / "retrieval_language_pair_matrix.csv", matrix_rows)

    # ---- (b) script blocks ----
    block_rows: list[dict] = []
    blocks = ["Euro<->Euro", "Euro<->ar", "Euro<->zh", "Ar<->Zh"]
    for cond in conditions:
        for model in MODELS + ["MEAN_OF_4_MODELS"]:
            for blk in blocks:
                vals = [float(r["recall_at_1"]) for r in matrix_rows
                        if r["condition"] == cond and r["model"] == model
                        and r["block"] == blk]
                if not vals:
                    continue
                block_rows.append({
                    "condition": cond, "model": model, "block": blk,
                    "n_ordered_pairs": len(vals),
                    "recall_at_1": f"{np.mean(vals):.4f}",
                    "times_chance": f"{np.mean(vals) / CHANCE:.1f}",
                    "min_pair": f"{min(vals):.4f}", "max_pair": f"{max(vals):.4f}",
                })
    write_csv(OUT_CSV_DIR / "retrieval_script_blocks.csv", block_rows)

    # ---- (c) per-language marginals, source vs target ----
    marg_rows: list[dict] = []
    for cond in conditions:
        for model in MODELS + ["MEAN_OF_4_MODELS"]:
            for lg in LANGS:
                def pick(key: str) -> list[float]:
                    return [float(r["recall_at_1"]) for r in matrix_rows
                            if r["condition"] == cond and r["model"] == model
                            and r[key] == lg]
                src_v, tgt_v = pick("src_lang"), pick("tgt_lang")
                if not src_v:
                    continue
                marg_rows.append({
                    "condition": cond, "model": model, "language": lg,
                    "as_source_recall_at_1": f"{np.mean(src_v):.4f}",
                    "as_target_recall_at_1": f"{np.mean(tgt_v):.4f}",
                    "averaged_over_both_directions": f"{np.mean(src_v + tgt_v):.4f}",
                    "n_pairs_each_direction": len(src_v),
                })
    write_csv(OUT_CSV_DIR / "retrieval_language_marginals.csv", marg_rows)

    # ---- printed report ----
    print(f"\nOperating point: normalized depth 0.5 (the paper's reported setting). "
          f"Chance Recall@1 = {CHANCE:.3f}.\n")
    for cond in conditions:
        print(f"=== {cond.upper()} — ordered-pair matrix, mean of 4 models "
              f"(rows = source, cols = target) ===")
        print("      " + "".join(f"{t:>8s}" for t in LANGS))
        for s in LANGS:
            cells = []
            for t in LANGS:
                if s == t:
                    cells.append(f"{'-':>8s}")
                    continue
                v = [r for r in matrix_rows if r["condition"] == cond
                     and r["model"] == "MEAN_OF_4_MODELS"
                     and r["src_lang"] == s and r["tgt_lang"] == t]
                cells.append(f"{float(v[0]['recall_at_1']):>8.3f}" if v else f"{'':>8s}")
            print(f"{s:>6s}" + "".join(cells))
        print()
        print(f"  {'block':14s}{'recall':>9s}{'x chance':>10s}{'min pair':>10s}{'max pair':>10s}")
        for blk in blocks:
            v = [r for r in block_rows if r["condition"] == cond
                 and r["model"] == "MEAN_OF_4_MODELS" and r["block"] == blk]
            if v:
                r0 = v[0]
                print(f"  {blk:14s}{float(r0['recall_at_1']):>9.3f}"
                      f"{float(r0['times_chance']):>10.1f}{float(r0['min_pair']):>10.3f}"
                      f"{float(r0['max_pair']):>10.3f}")
        print()

    print("=== per-language marginals, DEFLATED, mean of 4 models ===")
    print(f"  {'lang':6s}{'as source':>11s}{'as target':>11s}{'both':>8s}")
    for lg in LANGS:
        v = [r for r in marg_rows if r["condition"] == "deflated"
             and r["model"] == "MEAN_OF_4_MODELS" and r["language"] == lg][0]
        print(f"  {lg:6s}{float(v['as_source_recall_at_1']):>11.3f}"
              f"{float(v['as_target_recall_at_1']):>11.3f}"
              f"{float(v['averaged_over_both_directions']):>8.3f}")

    arzh = [r for r in block_rows if r["condition"] == "deflated"
            and r["model"] == "MEAN_OF_4_MODELS" and r["block"] == "Ar<->Zh"][0]
    euro = [r for r in block_rows if r["condition"] == "deflated"
            and r["model"] == "MEAN_OF_4_MODELS" and r["block"] == "Euro<->Euro"][0]
    arzh_per_model = {m: [r["recall_at_1"] for r in block_rows
                          if r["condition"] == "deflated" and r["model"] == m
                          and r["block"] == "Ar<->Zh"][0] for m in MODELS}



if __name__ == "__main__":
    main()
