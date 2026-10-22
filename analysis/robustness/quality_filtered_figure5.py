"""
quality_filtered_figure5.py -- Figure 5 recomputed on the quality-filtered
subset (FULL16, QC, QC12, RAND12), with the plotting code of
analysis/paper_figures/fig_image5_mixed_paper.py imported unchanged.

Top-5 arm-deflated log-volume (mean pooling) and raw participation ratio
(last token) are recomputed from the activations for each condition; the
FULL16 condition must reproduce the paper's figure.

Output
------
  results/robustness/quality_filtered_cache/{condition}/   (per-condition caches)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quality_filtered_rerun import (
    ACTS, K_DEFLATE, N_MATCHED, build_index, is_defective, is_truncated, has_cue,
)

ROOT = Path(__file__).resolve().parents[2]
FIG_SRC = ROOT / "analysis" / "paper_figures" / "fig_image5_mixed_paper.py"
OUT_CACHE = Path(__file__).resolve().parents[2] / "results" / "robustness" / "quality_filtered_cache"
OUT_IMG = ROOT / "results" / "figures_robustness"
ARMS = ["refinement", "commonality"]
LEVELS = [0, 1, 2, 3, 4]
ZY_SEED = 42            # analysis/utils.py:SEED, used by the published ZY cache


# --------------------------------------------------------------- selections
def make_selection(ds, condition: str):
    """{(chain, level): jsonl indices} for the given condition.

    RAND12 is the control that makes QC12 interpretable: both panels shrink when
    n falls from 16 to 12 for reasons that have nothing to do with sentence
    quality (fewer eigenvalues in the log-volume sum, a lower ceiling on PR).
    Comparing QC12 against FULL16 alone would show that shrinkage and invite it
    to be read as a quality effect. QC12 against RAND12 is the like-for-like test.
    """
    cells = {}
    for j, r in enumerate(ds):
        cells.setdefault((r["chain_id"], r["level"]), []).append(j)
    rng = np.random.default_rng(20260729)
    out = {}
    for k, v in sorted(cells.items()):
        if condition == "FULL16":
            sel = v
        elif condition == "RAND12":
            sel = sorted(rng.choice(v, N_MATCHED, replace=False).tolist())
        else:
            sel = [j for j in v if not is_defective(ds[j]["sentence"])]
            if condition == "QC12":
                sel = sel[:N_MATCHED]
        out[k] = np.array(sel)
    return out


# --------------------------------------------------------------- estimators
def log_volume_from_eigs(eig: np.ndarray, rank: int = 128) -> float:
    """Matches analysis/utils.py:log_volume for the 1-D (eigenvalue) branch."""
    lam = np.maximum(eig[:rank], 1e-30)
    return float(np.sum(np.log(lam)))


def build_lv_cache(model, ds, rows, sel, arm):
    """chain_lv (n_chains, L, 5): top-5 arm-deflated log-volume, mean pooling."""
    acts = np.load(str(ACTS / model / f"{model}_constraints_mean.npy"), mmap_mode="r")
    L = acts.shape[1]
    chains = sorted({r["chain_id"] for r in ds if r["arm"] == arm})
    out = np.full((len(chains), L, 5), np.nan)

    arm_rows = np.concatenate([sel[(c, l)] for c in chains for l in LEVELS])
    for layer in range(L):
        X = np.asarray(acts[rows[arm_rows], layer, :], dtype=np.float64)
        pos = {j: i for i, j in enumerate(arm_rows)}

        # Identical call to analysis/ZY_top5_deflated.py:_precompute_deflation --
        # randomized_svd with the project SEED, not an exact decomposition. An
        # exact top-5 basis is a slightly different subspace and shifts every
        # log-volume by ~0.45, which cancels in the S0->Sk gap but stops the
        # rebuild reproducing the published cache. Reproduce, do not improve.
        mu = X.mean(axis=0)
        Xc = X - mu
        _, _, Vt = randomized_svd(Xc, n_components=min(K_DEFLATE, len(Xc) - 1),
                                  random_state=ZY_SEED)
        V = Vt.T                                       # (D, K)

        for ci, ch in enumerate(chains):
            for li in LEVELS:
                idx = np.array([pos[j] for j in sel[(ch, li)]])
                if len(idx) < 3:
                    continue
                Z = X[idx] - mu
                Z = Z - (Z @ V) @ V.T
                Zc = Z - Z.mean(axis=0, keepdims=True)
                s = np.linalg.svd(Zc, compute_uv=False)
                eig = (s ** 2) / max(len(idx) - 1, 1)
                out[ci, layer, li] = log_volume_from_eigs(eig)
    return out


def build_pr_cache(model, ds, rows, sel, arm):
    """chain_pr (n_chains, L, 5): RAW participation ratio, last-token pooling."""
    acts = np.load(str(ACTS / model / f"{model}_constraints_last.npy"), mmap_mode="r")
    L = acts.shape[1]
    chains = sorted({r["chain_id"] for r in ds if r["arm"] == arm})
    out = np.full((len(chains), L, 5), np.nan)

    for ci, ch in enumerate(chains):
        for li in LEVELS:
            idx = sel[(ch, li)]
            if len(idx) < 2:
                continue
            X = np.asarray(acts[rows[idx], :, :], dtype=np.float64)   # (n, L, D)
            X = X.transpose(1, 0, 2)                                   # (L, n, D)
            Xc = X - X.mean(axis=1, keepdims=True)
            S = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
            S2 = S ** 2
            num, den = S2.sum(axis=1) ** 2, (S2 ** 2).sum(axis=1)
            out[ci, :, li] = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    return out


# --------------------------------------------------------------- figure
def render(condition: str, model: str, tag: str):
    """Import the paper's Figure 5 module and point it at this condition's cache."""
    spec = importlib.util.spec_from_file_location(f"_fig5_{condition}", FIG_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod._ZY_CACHE = OUT_CACHE / condition
    mod._B1_LAST = OUT_CACHE / condition
    mod.MODEL = model
    fig = mod.build_figure()
    fig.suptitle(tag, fontsize=11, y=1.02)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_IMG / f"Image_5_mixed_{condition}.{ext}"
        fig.savefig(str(p), bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"    saved -> {p.relative_to(ROOT)}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--conditions", nargs="+", default=["FULL16", "QC", "QC12", "RAND12"])
    args = ap.parse_args()

    ds, rows, _ = build_index()
    n_def = sum(1 for r in ds if is_defective(r["sentence"]))
    print(f"model {args.model}   defective {n_def}/{len(ds)} "
          f"(truncated {sum(1 for r in ds if is_truncated(r['sentence']))}, "
          f"cue {sum(1 for r in ds if has_cue(r['sentence']))})\n")

    published = None
    for cond in args.conditions:
        sel = make_selection(ds, cond)
        ns = [len(v) for v in sel.values()]
        print(f"  [{cond}] cells {len(sel)}  n/cell min={min(ns)} max={max(ns)} "
              f"median={int(np.median(ns))}")
        d = OUT_CACHE / cond
        d.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            lv = build_lv_cache(args.model, ds, rows, sel, arm)
            np.savez(d / f"{args.model}_{arm}_mean_id_lv.npz",
                     chain_lv=lv, chain_id=np.full_like(lv, np.nan))
            pr = build_pr_cache(args.model, ds, rows, sel, arm)
            np.savez(d / f"{args.model}_{arm}_raw.npz",
                     mean_pr=np.nanmean(pr, axis=0), sd_pr=np.nanstd(pr, axis=0, ddof=1),
                     chain_pr=pr)
            print(f"    {arm:12s} lv {lv.shape}  pr {pr.shape}")

        # VALIDATION: our FULL16 rebuild must match the published cache
        if cond == "FULL16":
            pub = ROOT / "results" / "caches" / "refinement" / "mean" / "ZY" / f"{args.model}_refinement_mean_id_lv.npz"
            if pub.exists():
                a = np.load(str(pub))["chain_lv"]
                b = np.load(str(d / f"{args.model}_refinement_mean_id_lv.npz"))["chain_lv"]
                m = np.isfinite(a) & np.isfinite(b)
                published = (float(np.corrcoef(a[m], b[m])[0, 1]),
                             float(np.max(np.abs(a[m] - b[m]))),
                             float(np.mean(np.abs(a[m] - b[m]))))
                print(f"    VALIDATION vs published cache: r={published[0]:.6f} "
                      f"max|diff|={published[1]:.4f} mean|diff|={published[2]:.4f}")

        label = {"FULL16": "Figure 5 — all 16 realisations (published pipeline, rebuilt)",
                 "QC": "Figure 5 — defective sentences removed (n = 12–16 per cell)",
                 "QC12": "Figure 5 — defective sentences removed, matched n = 12 per cell",
                 "RAND12": "Figure 5 — CONTROL: random 12 of 16, no quality filtering"}[cond]
        render(cond, args.model, f"{args.model}   {label}")
        print()



if __name__ == "__main__":
    main()
