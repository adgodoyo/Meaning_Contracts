"""
mahalanobis_rank_sensitivity.py -- sensitivity of the Mahalanobis ordering to
the retained rank r (Table 9, Appendix E).

The top-r eigenpairs of the layer covariance are computed exactly
(np.linalg.eigh on the D x D covariance). For each model, readout and
condition (raw, the figure deflation level, top-5) and each r, the script
records the mean truncated Mahalanobis distance per category, the Spearman rho
of the seven-category ordering against an r=512 reference, and whether the
core-and-spread ordering B holds (column three_band_ok). Layers: normalised
depth 0.5 and the stable-band midpoint. At r = N-1 whitening maps the centred
points onto a regular simplex and all categories collapse to one value.

Outputs
-------
  results/robustness/rank_sensitivity_layers.csv
  results/robustness/rank_sensitivity_summary.csv
  results/figures_robustness/rank_sensitivity.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FC_TABLES, FINAL_CODEBASE, MODELS, OUT_CSV_DIR, OUT_FIG_DIR, SEED,
    internal_to_paper, load_kmin, load_stable_band,
    read_csv_rows, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    PAIR_CATEGORIES, load_activations, load_multilingual_labels,
    build_multilingual_pair_indices, pairs_to_array,
    normalised_depth_array, top_k_deflation,
)

CATS = list(PAIR_CATEGORIES.keys())
PAPER_CATS = [f"C{i}" for i in range(1, 8)]

RANKS = [16, 32, 64, 128, 256, 384, 512, 768, 1024, 1199]
REF_RANK = 512
TARGET_ND = 0.5

MODEL_COLORS = {
    "pythia-2.8b": "#1f77b4",
    "llama3.2-3b": "#ff7f0e",
    "qwen2.5-3b":  "#2ca02c",
    "gemma4-e4b":  "#d62728",
}


# ---------------------------------------------------------------------------
# Exact truncated Mahalanobis
# ---------------------------------------------------------------------------

def exact_eigenspectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact eigendecomposition of the sample covariance of X, descending.

    Returns (vecs, vals) with vecs of shape (D, D) and vals of shape (D,).
    Uses eigh on the DxD covariance: symmetric, D <= 3072, exact to machine
    precision. No randomization, so the result is reproducible and is the
    ground truth against which the two released randomized sweeps are checked.
    """
    X64 = X.astype(np.float64)
    Xc = X64 - X64.mean(axis=0)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    vals, vecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(vals)[::-1]
    return vecs[:, order], vals[order]


def mahal_means(X: np.ndarray, pairs: dict, vecs: np.ndarray, vals: np.ndarray,
                rank: int) -> dict[str, float]:
    """Mean truncated-Mahalanobis distance per paper category at the given rank."""
    V = vecs[:, :rank]
    inv_sqrt = 1.0 / np.sqrt(np.maximum(vals[:rank], 1e-10))
    X64 = X.astype(np.float64)
    Xc = X64 - X64.mean(axis=0)
    X_pc = (Xc @ V) * inv_sqrt[None, :]
    out: dict[str, float] = {}
    for c in CATS:
        p = pairs[c]
        if len(p) == 0:
            out[internal_to_paper(c)] = np.nan
            continue
        diff = X_pc[p[:, 0]] - X_pc[p[:, 1]]
        out[internal_to_paper(c)] = float(np.linalg.norm(diff, axis=1).mean())
    return out


def three_band_ok(v: dict[str, float]) -> bool:
    """C1 nearest; C4..C7 middle; C2,C3 farthest — strict, no ties allowed."""
    mid = [v[c] for c in ("C4", "C5", "C6", "C7")]
    far = [v["C2"], v["C3"]]
    return v["C1"] < min(mid) and max(mid) < min(far)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def conditions_for(model: str, pooling: str) -> list[tuple[str, int]]:
    """(label, k) pairs; k=0 means raw. k_min is the paper's reported level."""
    kmin = load_kmin(model, pooling)
    conds = [("raw", 0), (f"top{kmin}_paper_kmin", kmin)]
    if kmin != 5:
        conds.append(("top5", 5))
    return conds


def run() -> list[dict]:
    labels = load_multilingual_labels()
    pair_lists = build_multilingual_pair_indices(labels)
    pairs = {c: pairs_to_array(pair_lists[c]) for c in CATS}
    n_sent = len(labels["domain"])

    rows: list[dict] = []
    for model in MODELS:
        for pooling in ("mean", "last"):
            acts = load_activations(model, "multilingual", pooling)
            N, L, D = acts.shape
            nd = normalised_depth_array(L)
            lo, hi = load_stable_band(model, pooling)
            layers = {
                "nd0.5": int(np.argmin(np.abs(nd - TARGET_ND))),
                "band_mid": (lo + hi - 1) // 2,
            }
            for layer_tag, layer in layers.items():
                X_raw = acts[:, layer, :].astype(np.float64)
                for cond_label, k in conditions_for(model, pooling):
                    X = X_raw if k == 0 else top_k_deflation(X_raw, k)
                    vecs, vals = exact_eigenspectrum(X)
                    max_rank = min(N - 1, D)
                    per_rank: dict[int, dict[str, float]] = {}
                    for R in RANKS:
                        if R > max_rank:
                            continue
                        per_rank[R] = mahal_means(X, pairs, vecs, vals, R)
                    ref = [per_rank[REF_RANK][c] for c in PAPER_CATS]
                    for R, vals_dict in per_rank.items():
                        cmp_vals = [vals_dict[c] for c in PAPER_CATS]
                        spread = max(cmp_vals) - min(cmp_vals)
                        if spread < 1e-9:
                            rho = np.nan      # degenerate: no ordering exists to correlate
                        else:
                            rho = float(spearmanr(cmp_vals, ref).statistic)
                        rows.append({
                            "model": model, "pooling": pooling,
                            "layer_tag": layer_tag, "layer": layer,
                            "condition": cond_label, "k": k, "rank": R,
                            "rho_vs_R512": "" if np.isnan(rho) else f"{rho:.4f}",
                            "three_band_ok": three_band_ok(vals_dict),
                            "category_spread": f"{spread:.6g}",
                            **{f"dM_{c}": f"{vals_dict[c]:.4f}" for c in PAPER_CATS},
                        })
                    print(f"  {model:12s} {pooling:4s} {layer_tag:8s} L{layer:2d} "
                          f"{cond_label:20s} rho@128="
                          f"{per_rank.get(128) and spearmanr([per_rank[128][c] for c in PAPER_CATS], ref).statistic:.4f}",
                          flush=True)

    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per model x pooling x condition (band-mid layer): where does the plateau start?"""
    out: list[dict] = []
    keys = sorted({(r["model"], r["pooling"], r["condition"]) for r in rows
                   if r["layer_tag"] == "band_mid"})
    for model, pooling, cond in keys:
        sub = sorted([r for r in rows
                      if r["layer_tag"] == "band_mid" and r["model"] == model
                      and r["pooling"] == pooling and r["condition"] == cond],
                     key=lambda r: r["rank"])
        rho = {r["rank"]: (float(r["rho_vs_R512"]) if r["rho_vs_R512"] else np.nan)
               for r in sub}
        band = {r["rank"]: r["three_band_ok"] for r in sub}
        finite = [(R, v) for R, v in sorted(rho.items()) if np.isfinite(v)]
        # plateau onset: smallest rank from which rho stays >= 0.95 for every larger finite rank
        onset = ""
        for i, (R, _) in enumerate(finite):
            if all(v >= 0.95 for _, v in finite[i:]):
                onset = R
                break
        out.append({
            "model": model, "pooling": pooling, "condition": cond,
            "rho_at_R128": f"{rho.get(128, np.nan):.4f}",
            "rho_at_R256": f"{rho.get(256, np.nan):.4f}",
            "rho_at_R1024": f"{rho.get(1024, np.nan):.4f}" if np.isfinite(rho.get(1024, np.nan)) else "",
            "plateau_onset_rank_rho_ge_0.95": onset,
            "three_band_ok_at_R128": band.get(128, ""),
            "three_band_ok_at_R256": band.get(256, ""),
            "three_band_ok_at_R512": band.get(512, ""),
            "degenerate_at_R1199": band.get(1199, ""),
        })
    return out


def make_figure(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, pooling in zip(axes, ("mean", "last")):
        for model in MODELS:
            sub = sorted([r for r in rows
                          if r["pooling"] == pooling and r["layer_tag"] == "band_mid"
                          and r["condition"].endswith("paper_kmin")
                          and r["model"] == model and r["rho_vs_R512"]],
                         key=lambda r: r["rank"])
            xs = [r["rank"] for r in sub]
            ys = [float(r["rho_vs_R512"]) for r in sub]
            ax.plot(xs, ys, "o-", color=MODEL_COLORS[model], label=model)
        reported = 128 if pooling == "mean" else 256
        ax.axvline(reported, color="k", linestyle="--", linewidth=1,
                   label=f"paper r={reported}")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Retained Mahalanobis rank r")
        ax.set_title(f"{pooling} pooling — paper deflation level, stable-band midpoint")
        ax.set_ylim(-1.05, 1.1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Spearman ρ of C1–C7 ordering vs r=512")
    fig.suptitle("Retained-rank sensitivity, exact eigendecomposition "
                 "(r=1199=N−1 omitted: all categories collapse identically)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT_FIG_DIR / "rank_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")


def main() -> None:
    rows = run()
    write_csv(OUT_CSV_DIR / "rank_sensitivity_layers.csv", rows)
    summary = summarize(rows)
    write_csv(OUT_CSV_DIR / "rank_sensitivity_summary.csv", summary)
    make_figure(rows)

    print("\n--- plateau summary (paper deflation level, stable-band midpoint) ---")
    for s in summary:
        if not s["condition"].endswith("paper_kmin"):
            continue
        print(f"  {s['model']:12s} {s['pooling']:4s}  rho@128={s['rho_at_R128']}  "
              f"rho@256={s['rho_at_R256']}  onset={s['plateau_onset_rank_rho_ge_0.95']}  "
              f"3band@128={s['three_band_ok_at_R128']} @256={s['three_band_ok_at_R256']}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
