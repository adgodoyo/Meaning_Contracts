"""
orderings_table.py -- LaTeX rows of Table 7 (range over the four models of the
percentage of layers satisfying strict R and B), from two_orderings_summary.csv.

Output
------
  results/robustness/orderings_table.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MODELS, OUT_CSV_DIR, read_csv_rows

METRICS = {"mean": [("cos", "Cosine"), ("eucl", "Euclidean"), ("mahal_r128", "Mahalanobis")],
           "last": [("cos", "Cosine"), ("eucl", "Euclidean"), ("mahal_r256", "Mahalanobis")]}
READOUT = {"mean": "Mean", "last": "Last-token"}


def span(values: list[float]) -> str:
    lo, hi = round(100 * min(values)), round(100 * max(values))
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def main() -> None:
    rows = read_csv_rows(OUT_CSV_DIR / "two_orderings_summary.csv")
    idx = {(r["model"], r["pooling"], r["metric"], r["condition"] == "raw"): r for r in rows}
    lines = []
    for pooling in ("mean", "last"):
        for mkey, mname in METRICS[pooling]:
            cells = []
            for col in ("frac_R", "frac_S"):
                raw = [float(idx[(m, pooling, mkey, True)][col]) for m in MODELS]
                fig = [float(idx[(m, pooling, mkey, False)][col]) for m in MODELS]
                cells.append(f"{span(raw)} $\\rightarrow$ {span(fig)}")
            lines.append(f"{READOUT[pooling]} & {mname} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    out = OUT_CSV_DIR / "orderings_table.tex"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
