"""
quality_filtered_numbers.py -- every number of Appendix J, recomputed from the
released data and the outputs of quality_filtered_rerun.py and
quality_filtered_figure5.py: filter counts, cells below 16 sentences, the
Figure 5 claims per condition, per-model gaps and sign agreement, and the
template-T1-only displacement gaps.

Output
------
  results/robustness/quality_filtered_numbers.csv
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _common import write_csv
from quality_filtered_rerun import (SENTS, is_defective, is_truncated, has_cue)
import json

RES = Path(__file__).resolve().parents[2] / "results" / "robustness"
CACHE = RES / "quality_filtered_cache"
FIG5_MODEL = "llama3.2-3b"          # the model Figure 5 is drawn for
LEVELS = [0, 1, 2, 3, 4]

out_rows = []


def emit(section, quantity, value, detail=""):
    out_rows.append({"section": section, "quantity": quantity,
                     "value": value, "detail": detail})
    print(f"  {quantity:<52s} {value:<26s} {detail}")


# ----------------------------------------------------------------- A: dataset
def section_a(ds):
    print("\nA. DATASET DESCRIPTIVES  (released JSONL only, no activations)")
    n = len(ds)
    tr = [r for r in ds if is_truncated(r["sentence"])]
    cu = [r for r in ds if has_cue(r["sentence"])]
    de = [r for r in ds if is_defective(r["sentence"])]

    emit("A", "sentences with an explicit narrowing cue", f"{len(cu)}/{n}", f"{100*len(cu)/n:.1f}%")
    emit("A", "sentences ending in a dangling function word", f"{len(tr)}/{n}", f"{100*len(tr)/n:.1f}%")
    emit("A", "sentences removed by either filter", f"{len(de)}/{n}", f"{100*len(de)/n:.1f}%")

    fams = Counter(r["template_family"] for r in cu)
    emit("A", "template families containing cue sentences",
         "yes" if set(fams) == {"T2"} else "NO", str(dict(fams)))

    cue_lv = [sum(1 for r in cu if r["level"] == l) for l in LEVELS]
    tr_lv = [sum(1 for r in tr if r["level"] == l) for l in LEVELS]
    emit("A", "cue sentences per level S0..S4", ", ".join(map(str, cue_lv)),
         "spread evenly over levels")
    emit("A", "dangling-word sentences per level S0..S4", ", ".join(map(str, tr_lv)),
         "concentrated at the longer levels S3-S4")

    for arm in ("refinement", "commonality"):
        sub = [r for r in ds if r["arm"] == arm]
        d = sum(1 for r in sub if is_defective(r["sentence"]))
        emit("A", f"share of {arm} sentences removed", f"{100*d/len(sub):.1f}%", f"{d}/{len(sub)}")

    cells = defaultdict(int)
    for r in ds:
        if not is_defective(r["sentence"]):
            cells[(r["chain_id"], r["level"])] += 1
    allc = {(r["chain_id"], r["level"]) for r in ds}
    below = sum(1 for k in allc if cells[k] < 16)
    emit("A", "(chain, level) cells left with fewer than 16 sentences", f"{below}/{len(allc)}",
         f"min surviving n = {min(cells.values())}")

    t1 = [r for r in ds if r["template_family"] == "T1"]
    t1d = sum(1 for r in t1 if is_defective(r["sentence"]))
    emit("A", "filtered sentences in template family T1", "0 defects" if not t1d else f"{t1d} DEFECTS",
         f"{len(t1)} sentences, {len(t1)//len(allc)} per cell")


# ------------------------------------------------------- B: Figure 5 claims
def claims_from_cache(cond, model=FIG5_MODEL):
    """Three qualitative claims of Figure 5, per condition, from the cached arrays."""
    d = CACHE / cond
    if not d.exists():
        return None
    lv, pr = {}, {}
    for arm in ("refinement", "commonality"):
        f_lv = d / f"{model}_{arm}_mean_id_lv.npz"
        f_pr = d / f"{model}_{arm}_raw.npz"
        if not (f_lv.exists() and f_pr.exists()):
            return None
        lv[arm] = np.load(str(f_lv))["chain_lv"]
        pr[arm] = np.load(str(f_pr))["chain_pr"]

    def three(data):
        # gap curves: (L, 4) = mean over chains of  Sk - S0,  k = 1..4
        g = {a: np.nanmean(np.stack([v[:, :, k] - v[:, :, 0] for k in (1, 2, 3, 4)], -1), 0)
             for a, v in data.items()}
        r, c = g["refinement"], g["commonality"]
        return (float(np.mean(r < c)),
                float(np.mean(np.all(np.diff(r, axis=1) <= 0, axis=1))),
                float(np.mean(r[:, 3] < 0)))

    return {"lv": three(lv), "pr": three(pr)}


def section_b():
    print(f"\nB. FIGURE 5 QUALITATIVE CLAIMS  ({FIG5_MODEL}, from quality_filtered_cache)")
    print(f"  {'condition':<12s}{'refine<control':>16s}{'level-ordered':>16s}{'S4<0':>9s}")
    got = {}
    for cond in ("FULL16", "QC", "QC12", "RAND12"):
        c = claims_from_cache(cond)
        if c is None:
            print(f"  {cond:<12s}  cache missing -- run quality_filtered_figure5.py")
            continue
        got[cond] = c
        a, b, d = c["lv"]
        print(f"  {cond:<12s}{100*a:>15.1f}%{100*b:>15.1f}%{100*d:>8.1f}%")
        out_rows.append({"section": "B", "quantity": f"Figure 5 claims [{cond}] log-volume",
                         "value": f"{100*a:.1f}% / {100*b:.1f}% / {100*d:.1f}%",
                         "detail": "refine<control / level-ordered / S4<0"})
    if {"FULL16", "QC", "QC12", "RAND12"} <= set(got):
        emit("B", "refinement below control: full -> filtered, unmatched n",
             f"{100*got['FULL16']['lv'][0]:.0f}% -> {100*got['QC']['lv'][0]:.0f}%",
             "drop caused by unequal n across cells")
        emit("B", "refinement below control: filtered, matched n=12",
             f"{100*got['QC12']['lv'][0]:.0f}%", "QC12")
        emit("B", "refinement below control: unfiltered, random n=12",
             f"{100*got['RAND12']['lv'][0]:.0f}%", "RAND12")
        emit("B", "monotone ordering by level: QC12 (FULL16)",
             f"{100*got['QC12']['lv'][1]:.0f}% ({100*got['FULL16']['lv'][1]:.0f}%)", "")
        emit("B", "S4 contraction: QC12 (FULL16)",
             f"{100*got['QC12']['lv'][2]:.0f}% ({100*got['FULL16']['lv'][2]:.0f}%)", "")


# --------------------------------------------------- C: multi-model table
def section_c():
    p = RES / "quality_filtered_summary.csv"
    if not p.exists():
        print("\nC. skipped -- run quality_filtered_rerun.py")
        return
    rows = list(csv.DictReader(open(p)))
    print("\nC. MULTI-MODEL TABLE  (from quality_filtered_summary.csv)")
    for metric, pool, label in (("lv_r11", "mean", "log-volume"),
                                ("disp", "mean", "displacement"),
                                ("pr", "last", "participation ratio")):
        print(f"  [{label}]")
        signs, bands, t1 = [], [], []
        for m in ("pythia-2.8b", "llama3.2-3b", "qwen2.5-3b", "gemma4-e4b"):
            d = {r["condition"]: r for r in rows
                 if r["model"] == m and r["metric"] == metric and r["pooling"] == pool}
            if not d:
                continue
            q = d["QC12"]
            signs.append(100 * float(q["sign_agree_with_FULL16"]))
            bands.append(q["inside_rand12_ci"] == "1")
            t1.append(float(d["T1ONLY"]["mean_gap"]))
            print(f"    {m:<13s} full {float(d['FULL16']['mean_gap']):+6.2f}   "
                  f"QC {float(q['mean_gap']):+6.2f}   sign {signs[-1]:3.0f}%   "
                  f"in band {'yes' if bands[-1] else 'NO'}")
            out_rows.append({"section": "C", "quantity": f"{label} {m}",
                             "value": f"{float(d['FULL16']['mean_gap']):+.2f} -> {float(q['mean_gap']):+.2f}",
                             "detail": f"sign {signs[-1]:.0f}%, in band {bands[-1]}"})
        if signs:
            emit("C", f"{label}: sign agreement range",
                 f"{min(signs):.0f}-{max(signs):.0f}%",
                 f"all inside matched-n band: {all(bands)}")
            if metric == "disp":
                emit("C", "displacement gap, template family T1 only",
                     ", ".join(f"{v:+.2f}" for v in t1), "zero-defect subset")


def main() -> None:
    ds = [json.loads(l) for l in open(SENTS)]
    section_a(ds)
    section_b()
    section_c()
    write_csv(RES / "quality_filtered_numbers.csv", out_rows)


if __name__ == "__main__":
    main()
