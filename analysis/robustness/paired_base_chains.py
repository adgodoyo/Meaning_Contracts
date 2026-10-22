"""
paired_base_chains.py -- refinement versus control on the same base noun
(Table 12, Appendix I).

Seven base nouns appear in both arms (certificate, form, invoice, license,
notice, record, report). Each refinement chain is compared with the control
chain built on the same noun, S0->S4, mean pooling, every layer: displacement
from the shared base, log-volume and participation ratio, plus the unpaired
(20 vs 20) comparison for reference.

Outputs
-------
  results/robustness/paired_base_chains_per_layer.csv
  results/robustness/paired_base_summary.csv
  results/robustness/paired_vs_unpaired.csv
  results/figures_robustness/paired_base_chains.png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FC_DATA, FINAL_CODEBASE, MODELS, OUT_CSV_DIR, OUT_FIG_DIR,
    build_constraints_activation_index, write_csv,
)

sys.path.insert(0, str(FINAL_CODEBASE / "analysis"))
from utils import (  # type: ignore
    load_activations, load_constraints_labels, log_volume, participation_ratio,
    normalised_depth_array,
)

REFINE_ARM = "refinement"
CONTROL_ARM = "commonality"
BASE_LEVEL, TOP_LEVEL = 0, 4


def find_paired_bases() -> list[tuple[str, str, str]]:
    """Return [(base_concept, refinement_chain_id, control_chain_id), ...]."""
    by_arm: dict[str, dict[str, str]] = defaultdict(dict)
    with open(FC_DATA / "constraints" / "constraints_chains.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            base = rec["levels"][0]["concept"]
            by_arm[rec["arm"]][base] = rec["chain_id"]
    shared = sorted(set(by_arm[REFINE_ARM]) & set(by_arm[CONTROL_ARM]))
    return [(b, by_arm[REFINE_ARM][b], by_arm[CONTROL_ARM][b]) for b in shared]


def chain_level_stats(acts: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For one (chain, level) cloud: return per-layer centroid (L, D),
    log-volume (L,) and participation ratio (L,).
    Mirrors final_codebase/analysis/B2_intrinsic_dim.py.
    """
    X = acts[mask].astype(np.float64)          # (n, L, D)
    n = X.shape[0]
    Xs = X.transpose(1, 0, 2)                  # (L, n, D)
    centroid = Xs.mean(axis=1)                 # (L, D)
    Xc = Xs - centroid[:, None, :]
    S = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)   # (L, n)
    eig = S ** 2 / max(n - 1, 1)               # (L, n)
    lv = np.array([log_volume(eig[l]) for l in range(eig.shape[0])])
    pr = np.array([participation_ratio(eig[l]) for l in range(eig.shape[0])])
    return centroid, lv, pr


def sign_test_p(n_pos: int, n: int) -> float:
    """Two-sided exact sign test p-value."""
    k = max(n_pos, n - n_pos)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> None:
    pairs = find_paired_bases()
    print(f"Paired base concepts found: {len(pairs)}")
    for b, r, c in pairs:
        print(f"   {b:12s}  refinement={r:5s}  control={c}")

    clab = load_constraints_labels()
    per_layer_rows, summary_rows, cmp_rows = [], [], []

    for model in MODELS:
        acts_full = load_activations(model, "constraints", "mean")
        idx = build_constraints_activation_index(model)
        acts = acts_full[idx]                      # aligned to the 3840 jsonl rows
        L = acts.shape[1]
        nd = normalised_depth_array(L)
        print(f"\n{model}: {acts.shape} (aligned), {L} layers")

        # ---- per-base paired quantities ---------------------------------
        disp = {}   # (base, arm) -> (L,)
        lvgap = {}
        prgap = {}
        for base, rid, cid in pairs:
            for arm, cid_ in ((REFINE_ARM, rid), (CONTROL_ARM, cid)):
                m0 = (clab["chain_id"] == cid_) & (clab["level"] == BASE_LEVEL)
                m4 = (clab["chain_id"] == cid_) & (clab["level"] == TOP_LEVEL)
                c0, lv0, pr0 = chain_level_stats(acts, m0)
                c4, lv4, pr4 = chain_level_stats(acts, m4)
                disp[(base, arm)] = np.linalg.norm(c4 - c0, axis=1)
                lvgap[(base, arm)] = lv4 - lv0
                prgap[(base, arm)] = pr4 - pr0

        # ---- per-layer paired contrasts ---------------------------------
        for li in range(L):
            d_diff = np.array([disp[(b, REFINE_ARM)][li] - disp[(b, CONTROL_ARM)][li] for b, _, _ in pairs])
            l_diff = np.array([lvgap[(b, REFINE_ARM)][li] - lvgap[(b, CONTROL_ARM)][li] for b, _, _ in pairs])
            p_diff = np.array([prgap[(b, REFINE_ARM)][li] - prgap[(b, CONTROL_ARM)][li] for b, _, _ in pairs])
            per_layer_rows.append({
                "model": model, "layer": li, "nd": round(float(nd[li]), 4),
                "n_bases": len(pairs),
                "disp_bases_refine_greater": int((d_diff > 0).sum()),
                "disp_mean_paired_diff": round(float(d_diff.mean()), 4),
                "logvol_bases_refine_more_contractive": int((l_diff < 0).sum()),
                "logvol_mean_paired_diff": round(float(l_diff.mean()), 4),
                "pr_bases_refine_lower": int((p_diff < 0).sum()),
                "pr_mean_paired_diff": round(float(p_diff.mean()), 4),
            })

        # ---- model-level summary ----------------------------------------
        sub = [r for r in per_layer_rows if r["model"] == model]
        n = len(pairs)
        for metric, key, direction in (
            ("displacement", "disp_bases_refine_greater", "refinement moves farther"),
            ("log_volume", "logvol_bases_refine_more_contractive", "refinement contracts more"),
            ("participation_ratio", "pr_bases_refine_lower", "refinement lower PR"),
        ):
            counts = np.array([r[key] for r in sub])
            diffs = np.array([r[metric.replace("displacement", "disp").replace("log_volume", "logvol")
                              .replace("participation_ratio", "pr") + "_mean_paired_diff"] for r in sub])
            layers_majority = int((counts > n / 2).sum())
            layers_unanimous = int((counts == n).sum())
            mean_count = float(counts.mean())
            summary_rows.append({
                "model": model, "metric": metric, "direction_tested": direction,
                "n_bases": n, "n_layers": L,
                "mean_bases_supporting": round(mean_count, 2),
                "layers_majority_pct": round(100 * layers_majority / L, 1),
                "layers_unanimous_pct": round(100 * layers_unanimous / L, 1),
                "mean_paired_diff": round(float(diffs.mean()), 4),
                "sign_test_p_at_mean_count": round(sign_test_p(int(round(mean_count)), n), 4),
            })

        # ---- paired (7 bases) vs unpaired (all 20 vs 20) -----------------
        for arm in (REFINE_ARM, CONTROL_ARM):
            chains = np.unique(clab["chain_id"][clab["arm"] == arm])
            d_all, l_all = [], []
            for cid_ in chains:
                m0 = (clab["chain_id"] == cid_) & (clab["level"] == BASE_LEVEL)
                m4 = (clab["chain_id"] == cid_) & (clab["level"] == TOP_LEVEL)
                c0, lv0, _ = chain_level_stats(acts, m0)
                c4, lv4, _ = chain_level_stats(acts, m4)
                d_all.append(np.linalg.norm(c4 - c0, axis=1))
                l_all.append(lv4 - lv0)
            cmp_rows.append({
                "model": model, "set": f"unpaired_all_{arm}", "n_chains": len(chains),
                "mean_displacement": round(float(np.mean(d_all)), 4),
                "mean_logvol_gap": round(float(np.mean(l_all)), 4),
            })
        for arm in (REFINE_ARM, CONTROL_ARM):
            cmp_rows.append({
                "model": model, "set": f"paired7_{arm}", "n_chains": len(pairs),
                "mean_displacement": round(float(np.mean([disp[(b, arm)] for b, _, _ in pairs])), 4),
                "mean_logvol_gap": round(float(np.mean([lvgap[(b, arm)] for b, _, _ in pairs])), 4),
            })

        del acts, acts_full

    write_csv(OUT_CSV_DIR / "paired_base_chains_per_layer.csv", per_layer_rows)
    write_csv(OUT_CSV_DIR / "paired_base_summary.csv", summary_rows)
    write_csv(OUT_CSV_DIR / "paired_vs_unpaired.csv", cmp_rows)

    # ---- printed summary -------------------------------------------------
    print("\n" + "=" * 78)
    print(f"PAIRED WITHIN-BASE CONTRAST  (n={len(pairs)} base concepts, mean pooling, S0->S4)")
    print("=" * 78)
    print(f"{'model':14s}{'metric':22s}{'mean bases':>12s}{'maj. layers':>13s}{'unanim.':>10s}{'mean diff':>12s}")
    for r in summary_rows:
        print(f"{r['model']:14s}{r['metric']:22s}{r['mean_bases_supporting']:>8.2f}/{r['n_bases']:<3d}"
              f"{r['layers_majority_pct']:>12.1f}%{r['layers_unanimous_pct']:>9.1f}%{r['mean_paired_diff']:>12.3f}")

    print("\nPaired-7 vs unpaired-20 (sanity: paired subset reproduces the main contrast)")
    print(f"{'model':14s}{'set':28s}{'mean disp':>12s}{'mean logvol gap':>18s}")
    for r in cmp_rows:
        print(f"{r['model']:14s}{r['set']:28s}{r['mean_displacement']:>12.3f}{r['mean_logvol_gap']:>18.3f}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, model in enumerate(MODELS):
        sub = [r for r in per_layer_rows if r["model"] == model]
        xs = [r["nd"] for r in sub]
        n = len(pairs)
        ax = axes[0, col]
        ax.plot(xs, [r["disp_bases_refine_greater"] for r in sub], color="#1f77b4", lw=1.8)
        ax.axhline(n / 2, color="grey", ls=":", lw=1)
        ax.set_ylim(0, n); ax.set_title(f"{model}\ndisplacement", fontsize=9)
        ax.set_xlabel("normalised depth"); ax.set_ylabel(f"bases w/ refine > control (of {n})")
        ax = axes[1, col]
        ax.plot(xs, [r["logvol_bases_refine_more_contractive"] for r in sub], color="#d62728", lw=1.8)
        ax.axhline(n / 2, color="grey", ls=":", lw=1)
        ax.set_ylim(0, n); ax.set_title("log-volume contraction", fontsize=9)
        ax.set_xlabel("normalised depth"); ax.set_ylabel(f"bases w/ refine more contractive (of {n})")
    fig.suptitle(f"Within-base paired refinement vs. control (n={len(pairs)} shared base concepts, S0→S4)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = OUT_FIG_DIR / "paired_base_chains.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out_png}")



if __name__ == "__main__":
    main()
