"""
Shared utilities for analysis/robustness/: paths, the mapping between the
analysis code's pair-category numbering and the paper's, CSV helpers, and the
row alignment between the refinement probe and its activation arrays.

Category numbering. analysis/utils.PAIR_CATEGORIES numbers the seven
cross-lingual pair categories in construction order; the paper swaps the first
two (paper C1 = exact translation = internal C2, paper C2 = internal C1).
Scripts convert with internal_to_paper() / paper_to_internal().
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]          # final_codebase root
FINAL_CODEBASE = ROOT
FC_TABLES = ROOT / "results" / "tables"
FC_ACTIVATIONS = ROOT / "results" / "activations"
FC_DATA = ROOT / "data"

RESULTS_TABLES = FC_TABLES
RESULTS_TABLES_LAST = FC_TABLES
RESULTS_CACHE = ROOT / "results" / "caches" / "refinement" / "mean"
RESULTS_CACHE_LAST = ROOT / "results" / "caches" / "refinement" / "last"

OUT_CSV_DIR = ROOT / "results" / "robustness"
OUT_FIG_DIR = ROOT / "results" / "figures_robustness"
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


MODELS = ["pythia-2.8b", "llama3.2-3b", "qwen2.5-3b", "gemma4-e4b"]
POOLINGS = ["mean", "last"]
SEED = 42

# ---------------------------------------------------------------------------
# Category-scheme remap (INTERNAL <-> PAPER)
# ---------------------------------------------------------------------------

# internal_label -> paper_label
INTERNAL_TO_PAPER = {
    "C1": "C2",  # same factor, same language
    "C2": "C1",  # exact translation (nearest / invariant core)
    "C3": "C3",  # same factor, diff language
    "C4": "C4",
    "C5": "C5",
    "C6": "C6",
    "C7": "C7",
}
PAPER_TO_INTERNAL = {v: k for k, v in INTERNAL_TO_PAPER.items()}

# Sanity: the remap must be an involution-free bijection on {C1..C7}
assert sorted(INTERNAL_TO_PAPER.keys()) == [f"C{i}" for i in range(1, 8)]
assert sorted(INTERNAL_TO_PAPER.values()) == [f"C{i}" for i in range(1, 8)]

PAPER_CATEGORY_DEFS = {
    "C1": "Exact meaning match, cross-language (parallel translation)",
    "C2": "Same factor, same language (independent realization)",
    "C3": "Same factor, cross-language (independent realization)",
    "C4": "Same domain, different factor, same language",
    "C5": "Same domain, different factor, cross-language",
    "C6": "Different domain, same language",
    "C7": "Different domain, cross-language",
}


def internal_to_paper(label: str) -> str:
    return INTERNAL_TO_PAPER[label]


def paper_to_internal(label: str) -> str:
    return PAPER_TO_INTERNAL[label]


def remap_dict_keys_internal_to_paper(d: dict) -> dict:
    """Given {internal_cat: value}, return {paper_cat: value}."""
    return {INTERNAL_TO_PAPER[k]: v for k, v in d.items() if k in INTERNAL_TO_PAPER}


def remap_order_internal_to_paper(order: list[str]) -> list[str]:
    """Given an ordering like ['C2','C4',...] in internal labels, relabel to paper scheme."""
    return [INTERNAL_TO_PAPER[c] for c in order]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(rows)} rows)")


def read_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# k_min (deflation level used in the paper's figures, per model)
# ---------------------------------------------------------------------------

KMIN_PAPER = {
    "pythia-2.8b": 8,
    "llama3.2-3b": 5,
    "qwen2.5-3b": 5,
    "gemma4-e4b": 5,
}


def load_kmin(model: str, pooling: str = "mean") -> int:
    col = "k_min" if pooling == "mean" else "k_min_last"
    path = FC_TABLES / "kmin_per_model.csv"
    if path.exists():
        for row in read_csv_rows(path):
            if row["model"] == model and row.get(col):
                return int(row[col])
    return KMIN_PAPER[model]


def load_stable_band(model: str, pooling: str = "mean") -> tuple[int, int]:
    path = FC_TABLES / ("stable_band_last.csv" if pooling == "last" else "stable_band.csv")
    for row in read_csv_rows(path):
        if row["model"] == model:
            return int(row["l_lo"]), int(row["l_hi"])
    raise KeyError(f"model {model} not found in {path}")


def build_constraints_activation_index(model: str) -> np.ndarray:
    """
    Row index of each refinement-probe sentence in the activation arrays.

    For each row of data/constraints/constraints_sentences.jsonl (in file
    order), returns the index of the matching row in
    results/activations/{model}/{model}_constraints_{pooling}.npy, using the
    sentence order stored in the extraction metadata:

      - metadata["constraints"]["sentence_order"] (sentence text, truncated to
        50 characters): matched in file order;
      - otherwise metadata["constraints"]["chain_ids"/"arms"/"levels"]:
        matched positionally within each (chain_id, arm, level) group.

    For activations extracted with extraction/extract_activations.py from the
    released data, the result is the identity.
    """
    import json as _json

    jsonl_path = FC_DATA / "constraints" / "constraints_sentences.jsonl"
    jsonl_records = []
    with open(jsonl_path) as f:
        for line in f:
            jsonl_records.append(_json.loads(line))

    meta_path = FC_ACTIVATIONS / model / f"{model}_metadata.json"
    with open(meta_path) as f:
        meta = _json.load(f)["constraints"]

    if "sentence_order" in meta:
        order_trunc = [s[:50] for s in meta["sentence_order"]]
        jsonl_trunc = [r["sentence"][:50] for r in jsonl_records]
        ptr = 0
        mapping = []
        for s in jsonl_trunc:
            found = None
            for j in range(ptr, len(order_trunc)):
                if order_trunc[j] == s:
                    found = j
                    break
            if found is None:
                raise ValueError(f"{model}: could not align sentence {s!r} past ptr={ptr}")
            mapping.append(found)
            ptr = found + 1
        return np.array(mapping, dtype=np.int64)

    # Fallback: positional matching within (chain_id, arm, level) groups.
    meta_keys = list(zip(meta["chain_ids"], meta["arms"], meta["levels"]))
    from collections import defaultdict
    group_positions: dict[tuple, list[int]] = defaultdict(list)
    for idx, key in enumerate(meta_keys):
        group_positions[key].append(idx)
    group_cursor: dict[tuple, int] = defaultdict(int)
    mapping = []
    for r in jsonl_records:
        key = (r["chain_id"], r["arm"], r["level"])
        cursor = group_cursor[key]
        if key not in group_positions or cursor >= len(group_positions[key]):
            raise ValueError(f"{model}: could not align record {r!r} (group {key} exhausted)")
        mapping.append(group_positions[key][cursor])
        group_cursor[key] += 1
    return np.array(mapping, dtype=np.int64)


if __name__ == "__main__":
    print("Category remap check:")
    for i in range(1, 8):
        internal = f"C{i}"
        paper = internal_to_paper(internal)
        print(f"  internal {internal}  ->  paper {paper}  ({PAPER_CATEGORY_DEFS[paper]})")
    print("\nModels:", MODELS)
    print("k_min:", {m: load_kmin(m) for m in MODELS})
