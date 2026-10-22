"""
quality_filtered_rerun.py -- the refinement analysis rerun on a quality-filtered
subset of the refinement probe (Appendix J).

Two mechanical filters, re-applicable to the released data:
  cue       sentences containing an explicit narrowing cue ("More narrowly",
            "More precisely", "Specifically", "Strictly", "Technically",
            "In precise terms")
  dangling  sentences ending in a dangling function word ("... is.")

Because log-volume and participation ratio are covariance quantities whose
range depends on n, filtered cells are compared at a matched n:
  FULL16   all 16 realisations
  QC12     filtered, then trimmed to 12 per (chain, level)
  RAND12   random 12 of 16, unfiltered (control for the n reduction)
  LOTO_Tk  one template family removed
  T1ONLY   template family T1 only (contains no filtered sentence)

Outputs
-------
  results/robustness/quality_filtered_summary.csv
  results/robustness/quality_filtered_raw_{model}_{pooling}.npy   (per-layer gaps)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import write_csv

ROOT = Path(__file__).resolve().parents[2]
FC = ROOT
ACTS = FC / "results" / "activations"
SENTS = FC / "data" / "constraints" / "constraints_sentences.jsonl"
OUT = Path(__file__).resolve().parents[2] / "results" / "robustness"

MODELS = ["pythia-2.8b", "llama3.2-3b", "qwen2.5-3b", "gemma4-e4b"]
ARMS = ["refinement", "commonality", "adv_near_synonym", "adv_scope_hedge"]
K_DEFLATE = 5          # matches analysis/03_logvolume_refinement.py
SEED = 42
N_MATCHED = 12         # observed minimum surviving cell size
LV_RANK = 11           # N_MATCHED - 1: usable eigenvalues at the matched size

# Defect filters. Mechanical, so that they can be re-applied to the released JSONL.
DANGLE = re.compile(r"\b(a|an|the|to|of|for|in|on|with|and|or|is|are|was|were|by|at|from)\.$", re.I)
CUES = ("more narrowly", "more precisely", "specifically", "strictly,",
        "technically,", "in precise terms")


def is_truncated(s: str) -> bool:
    return bool(DANGLE.search(s))


def has_cue(s: str) -> bool:
    return any(c in s.lower() for c in CUES)


def is_defective(s: str) -> bool:
    return is_truncated(s) or has_cue(s)


# ------------------------------------------------------------------ alignment
def build_index():
    """Map each released JSONL record to its activation row.

    Both orderings group sentences into (chain, level) cells of 16 in the same
    order, so the mapping is positional within a cell. Verified against the
    50-character truncated strings in the metadata before use.
    """
    ds = [json.loads(l) for l in open(SENTS)]
    meta = json.load(open(ACTS / "pythia-2.8b" / "pythia-2.8b_metadata.json"))["constraints"]
    ch, lv, so = meta["chain_ids"], meta["levels"], meta["sentence_order"]

    acells, jcells = collections.defaultdict(list), collections.defaultdict(list)
    for i, (c, l) in enumerate(zip(ch, lv)):
        acells[(c, l)].append(i)
    for j, r in enumerate(ds):
        jcells[(r["chain_id"], r["level"])].append(j)

    rows = np.full(len(ds), -1, dtype=np.int64)
    checked = 0
    for k, js in jcells.items():
        a = acells[k]
        if len(a) != len(js):
            sys.exit(f"cell size mismatch at {k}: {len(a)} acts vs {len(js)} jsonl")
        for ai, ji in zip(a, js):
            if not ds[ji]["sentence"].startswith(so[ai]):
                sys.exit(f"alignment check failed at {k}: {so[ai]!r} vs {ds[ji]['sentence']!r}")
            rows[ji] = ai
            checked += 1
    assert (rows >= 0).all()

    # every model must share the row ordering, or the mapping is model-specific
    for m in MODELS[1:]:
        mm = json.load(open(ACTS / m / f"{m}_metadata.json"))["constraints"]
        if mm["chain_ids"] != ch or mm["levels"] != lv:
            sys.exit(f"{m} activation ordering differs from pythia-2.8b")
    return ds, rows, checked


# ------------------------------------------------------------------ conditions
def build_conditions(ds, repeats: int):
    """{name: {(chain, level): np.array of JSONL indices}}."""
    cells = collections.defaultdict(list)
    for j, r in enumerate(ds):
        cells[(r["chain_id"], r["level"])].append(j)
    cells = {k: np.array(v) for k, v in cells.items()}
    rng = np.random.default_rng(SEED)
    cond = {}

    cond["FULL16"] = {k: v.copy() for k, v in cells.items()}

    # QC: drop defective, then trim to the matched size (drop from the end so the
    # choice is deterministic and does not depend on the RNG stream)
    qc = {}
    for k, v in cells.items():
        clean = np.array([j for j in v if not is_defective(ds[j]["sentence"])])
        if len(clean) < N_MATCHED:
            sys.exit(f"cell {k} has only {len(clean)} clean sentences (< {N_MATCHED})")
        qc[k] = clean[:N_MATCHED]
    cond["QC12"] = qc

    for r in range(repeats):
        cond[f"RAND12_{r:02d}"] = {k: rng.choice(v, N_MATCHED, replace=False)
                                   for k, v in cells.items()}

    for fam in ["T1", "T2", "T3", "T4"]:
        cond[f"LOTO_{fam}"] = {
            k: np.array([j for j in v if ds[j]["template_family"] != fam])
            for k, v in cells.items()}

    cond["T1ONLY"] = {k: np.array([j for j in v if ds[j]["template_family"] == "T1"])
                      for k, v in cells.items()}
    return cond, cells


# ------------------------------------------------------------------ metrics
def cell_metrics(X: np.ndarray):
    """X: (n, D) deflated. Returns (lv_full, lv_r11, pr, centroid).

    The centroid is returned so displacement can be measured centroid-to-centroid
    against the chain's own level-0 cell. Unlike log-volume and PR -- which are
    covariance quantities and so degrade sharply as n falls -- a centroid is well
    estimated from very few points, which is what makes the zero-defect T1-only
    condition (n=4) interpretable for displacement and not for the other two.
    """
    n = X.shape[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    s = np.linalg.svd(Xc, compute_uv=False)
    eig = np.maximum((s ** 2) / max(n - 1, 1), 0.0)
    lv_full = float(np.sum(np.log(np.maximum(eig, 1e-30))))
    lv_r11 = float(np.sum(np.log(np.maximum(np.sort(eig)[::-1][:LV_RANK], 1e-30))))
    tr, tr2 = float(eig.sum()), float((eig ** 2).sum())
    pr = (tr ** 2) / tr2 if tr2 > 1e-30 else 0.0
    return lv_full, lv_r11, pr, mu


def run_model(model: str, pooling: str, ds, rows, cond, cells, verbose=True):
    path = ACTS / model / f"{model}_constraints_{pooling}.npy"
    acts = np.load(str(path), mmap_mode="r")
    L = acts.shape[1]
    arm_of = np.array([r["arm"] for r in ds])
    chain_of = np.array([r["chain_id"] for r in ds])
    level_of = np.array([r["level"] for r in ds])

    keys = sorted(cells)
    recs = []
    for l in range(L):
        # one disk read per layer, for the released rows only
        Xl = np.asarray(acts[rows, l, :], dtype=np.float64)

        # top-5 deflation basis per arm, computed once from the FULL data so that
        # filtering changes the clouds and not the basis they are measured in
        defl = {}
        for arm in ARMS:
            idx = np.where(arm_of == arm)[0]
            Xa = Xl[idx]
            mu = Xa.mean(axis=0)
            Xc = Xa - mu
            # exact top-k via the Gram matrix (n << D), deterministic
            G = Xc @ Xc.T
            w, U = np.linalg.eigh(G)
            U = U[:, ::-1][:, :K_DEFLATE]
            w = np.maximum(w[::-1][:K_DEFLATE], 1e-30)
            V = (Xc.T @ U) / np.sqrt(w)          # (D, K)
            defl[arm] = (mu, V)

        for cname, sel in cond.items():
            stash, base = {}, {}
            for (chain, level) in keys:
                jidx = sel[(chain, level)]
                if len(jidx) < 3:
                    continue
                arm = arm_of[jidx[0]]
                mu, V = defl[arm]
                Xd = Xl[jidx] - mu
                Xd = Xd - (Xd @ V) @ V.T
                lv_f, lv_r, pr, cen = cell_metrics(Xd)
                stash[(chain, level)] = (arm, len(jidx), lv_f, lv_r, pr, cen)
                if level == 0:
                    base[chain] = cen
            for (chain, level), (arm, n, lv_f, lv_r, pr, cen) in stash.items():
                disp = (float(np.linalg.norm(cen - base[chain]))
                        if chain in base else np.nan)
                recs.append((cname, chain, arm, level, l, n, lv_f, lv_r, pr, disp))
        if verbose and (l % 8 == 0 or l == L - 1):
            print(f"    layer {l+1}/{L}", flush=True)

    dt = np.dtype([("cond", "U12"), ("chain", "U8"), ("arm", "U18"), ("level", "i4"),
                   ("layer", "i4"), ("n", "i4"), ("lv_full", "f8"), ("lv_r11", "f8"),
                   ("pr", "f8"), ("disp", "f8")])
    return np.array(recs, dtype=dt)


# ------------------------------------------------------------------ summarise
def gap_curve(rec, cond, metric, hi=4, lo=0):
    """Per-layer refinement-minus-commonality gap, matching Figure 5.

    The paper differences WITHIN each chain first (analysis/03, `_gap_exact`:
    chain[:, :, hi] - chain[:, :, lo]), averages that over the 20 chains of an
    arm, then contrasts the two arms. Differencing within chain removes each
    chain's own baseline, so the result measures contraction along the chain
    rather than differences between the base concepts.
    """
    m = rec[rec["cond"] == cond]
    layers = np.unique(m["layer"])
    out = np.full(len(layers), np.nan)
    for i, l in enumerate(layers):
        s = m[m["layer"] == l]
        per_arm = {}
        for arm in ("refinement", "commonality"):
            a = s[s["arm"] == arm]
            d = []
            for chain in np.unique(a["chain"]):
                ac = a[a["chain"] == chain]
                vh = ac[ac["level"] == hi][metric]
                vl = ac[ac["level"] == lo][metric]
                if len(vh) and len(vl):
                    d.append(vh[0] - vl[0])
            per_arm[arm] = np.mean(d) if d else np.nan
        out[i] = per_arm["refinement"] - per_arm["commonality"]
    return out


def summarise(model, pooling, rec, repeats):
    rows = []
    rand = [f"RAND12_{r:02d}" for r in range(repeats)]
    for metric in ["lv_full", "lv_r11", "pr", "disp"]:
        ref = gap_curve(rec, "FULL16", metric)
        rc = np.vstack([gap_curve(rec, c, metric) for c in rand]) if rand else None
        for cname in ["FULL16", "QC12", "LOTO_T1", "LOTO_T2", "LOTO_T3", "LOTO_T4", "T1ONLY"]:
            g = gap_curve(rec, cname, metric)
            ok = np.isfinite(g) & np.isfinite(ref)
            r_pearson = float(np.corrcoef(g[ok], ref[ok])[0, 1]) if ok.sum() > 2 else np.nan
            row = {"model": model, "pooling": pooling, "metric": metric, "condition": cname,
                   "n_per_cell": int(np.median(rec[rec["cond"] == cname]["n"])),
                   "mean_gap": round(float(np.nanmean(g)), 4),
                   "corr_with_FULL16": round(r_pearson, 4),
                   "sign_agree_with_FULL16": round(float(np.mean(np.sign(g[ok]) == np.sign(ref[ok]))), 4)}
            if rc is not None:
                lo, hi = np.nanpercentile(np.nanmean(rc, axis=1), [2.5, 97.5])
                row["rand12_mean_gap"] = round(float(np.nanmean(rc)), 4)
                row["rand12_ci_lo"] = round(float(lo), 4)
                row["rand12_ci_hi"] = round(float(hi), 4)
                row["inside_rand12_ci"] = int(lo <= np.nanmean(g) <= hi)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--poolings", nargs="+", default=["mean", "last"])
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    ds, rows, checked = build_index()
    print(f"alignment verified: {checked}/{len(ds)} rows matched positionally "
          f"and confirmed against metadata prefixes")

    n_def = sum(1 for r in ds if is_defective(r["sentence"]))
    print(f"defective sentences: {n_def}/{len(ds)} ({100*n_def/len(ds):.1f}%)  "
          f"[truncated {sum(1 for r in ds if is_truncated(r['sentence']))}, "
          f"cue {sum(1 for r in ds if has_cue(r['sentence']))}]")

    cond, cells = build_conditions(ds, args.repeats)
    print(f"conditions: {len([c for c in cond if not c.startswith('RAND')])} fixed "
          f"+ {args.repeats} random draws\n")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for model in args.models:
        for pooling in args.poolings:
            if not (ACTS / model / f"{model}_constraints_{pooling}.npy").exists():
                print(f"  [skip] {model}/{pooling}: activations not found")
                continue
            print(f"  {model} / {pooling}")
            rec = run_model(model, pooling, ds, rows, cond, cells)
            np.save(OUT / f"quality_filtered_raw_{model}_{pooling}.npy", rec)
            summary += summarise(model, pooling, rec, args.repeats)

    write_csv(OUT / "quality_filtered_summary.csv", summary)

    print(f"\n{'model':14s}{'pool':6s}{'metric':9s}{'condition':10s}{'n':>4}"
          f"{'gap':>10}{'corr':>8}{'in CI':>7}")
    for r in summary:
        if r["metric"] == "lv_r11":
            print(f"{r['model']:14s}{r['pooling']:6s}{r['metric']:9s}{r['condition']:10s}"
                  f"{r['n_per_cell']:>4}{r['mean_gap']:>10.3f}{r['corr_with_FULL16']:>8.3f}"
                  f"{r.get('inside_rand12_ci','-'):>7}")



if __name__ == "__main__":
    main()
