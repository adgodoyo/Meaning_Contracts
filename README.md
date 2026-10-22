# Meaning Contracts: Local Semantic Geometry in Large Language Models

Code and data for the paper *Meaning Contracts: Local Semantic Geometry in Large
Language Models* (Andrés Daniel Godoy Ortiz, Imperial College London).

The paper studies the local geometry of meaning in the hidden states of four
decoder-only LLMs (Pythia 2.8B, Llama 3.2 3B, Qwen 2.5 3B, Gemma 4 E4B) using two
controlled probes: a cross-lingual semantic-neighbourhood probe and a semantic
refinement probe. All analyses are training-free: hidden states are extracted
once from frozen models and compared across layers, readouts and conditions.

## Repository layout

```
data/
  multilingual/multilingual.jsonl          cross-lingual probe: 40 factors x 5 languages x 6 sentences (1,200)
  constraints/constraints_chains.jsonl      refinement probe: 48 chains
  constraints/constraints_sentences.jsonl   refinement probe: 3,840 sentences (chain, arm, level, template, variant)
extraction/
  extract_activations.py                    hidden states for both probes, every layer, mean and last-token readout
  extract_position_eigvals.py               per-token-position covariance spectra (Figure 6)
notebooks/
  extract_hidden_states.ipynb               the same extraction for a Colab GPU
analysis/
  utils.py                                  loaders, pair construction, deflation, distances, spread measures
  01-15_*.py                                main analyses (caches and tables)
  robustness/                               the analyses behind Appendices D-N and the two orderings
  paper_figures/                            every figure of the paper
results/
  caches/                                   intermediate arrays; figures and tables can be rebuilt from them
  tables/                                   tables of the main analyses
  robustness/                               outputs of analysis/robustness/
  paper_figures/                            the figures as they appear in the paper
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.12
pip install -r requirements.txt
```

Llama 3.2 is a gated model: accept its licence on Hugging Face and set `HF_TOKEN`
in the environment (or in a local `.env` file, which is git-ignored) before
extracting.

## Reproducing without the activations

The hidden-state activations (~15 GB) are not included; `results/caches/` is.
These scripts run directly after cloning:

```bash
python analysis/paper_figures/make_paper_figures.py           # Figures 1-8 -> results/paper_figures/
python analysis/10_appendix_tables.py                         # Tables 4, 15, 16
python analysis/12_adversarial_tables.py                      # adversarial peak gaps (Section 4.2)
python analysis/15_displacement_table.py                      # Table 14
python analysis/robustness/orderings_table.py                 # Table 7
python analysis/robustness/retrieval_language_summary.py      # Table 10
python analysis/robustness/quality_filtered_numbers.py        # Appendix J
python analysis/robustness/token_counts_by_level.py           # Table 17
```

Every other script reads the activations and needs the extraction step first.

## Full pipeline

**1. Extract hidden states** (GPU or Apple MPS, bfloat16, saved as float32).

```bash
python extraction/extract_activations.py --model pythia-2.8b
python extraction/extract_activations.py --model llama3.2-3b
python extraction/extract_activations.py --model qwen2.5-3b
python extraction/extract_activations.py --model gemma4-e4b --batch-size 1
python extraction/extract_position_eigvals.py
```

This writes `results/activations/{model}/{model}_{multilingual,constraints}_{mean,last}.npy`
with shape (sentences, layers including the embedding layer, hidden width).

**2. Main analyses** (caches in `results/caches/`, tables in `results/tables/`).

| Script | Computes | Paper |
|---|---|---|
| `01_multilingual_geometry.py` | cosine, Euclidean and Mahalanobis category profiles, raw and top-k deflated | Figures 1–3 |
| `02_refinement_distances.py` | S0→Sk displacement, refinement vs control; token counts | Figure 4; Tables 14, 17 |
| `03_logvolume_refinement.py` | log-volume of chain-level clouds, top-5 deflated | Figure 5; Table 15 |
| `04_participation_ratio.py` | participation ratio of chain-level clouds | Figure 5; Table 16 |
| `05_position_analysis.py` | per-position log-volume increments | Section 4.2, Figure 6 |
| `10_appendix_tables.py` | log-volume, participation-ratio and facet tables; k per model | Tables 4, 15, 16 |
| `11_retrieval_analysis.py` | cross-lingual Recall@1, raw vs deflated | Section 4.1 |
| `12_adversarial_tables.py`, `13_adversarial_profiles.py` | near-synonym and scope-hedge controls | Section 4.2; Figures 7, 8 |
| `14_mahalanobis_profiles.py` | per-layer Mahalanobis and Euclidean category profiles | Figure 2 |
| `15_displacement_table.py` | refinement-minus-control displacement summary | Table 14 |

**3. Robustness analyses** (`analysis/robustness/`, outputs in `results/robustness/`).

| Script | Paper |
|---|---|
| `two_orderings.py` → `orderings_table.py` | Section 3.2, Section 4.1, Tables 7 and 8 |
| `euclidean_deflation_sweep.py` | Table 13, Appendix K |
| `mahalanobis_rank_sensitivity.py` | Table 9, Appendix E |
| `covariance_by_depth.py` | Appendix F |
| `retrieval_language_pairs.py` → `retrieval_language_summary.py` | Table 10, Appendix G |
| `retrieval_depth_sweep.py`, `retrieval_layer_sweep.py`, `retrieval_layer_sweep_last_token.py` | Section 4.1, Appendix G |
| `spread_sample_size.py`, `spread_permutation_and_calibration.py`, `spread_centering_eigenvalue.py` | Section 3.3, Table 11, Appendix H |
| `paired_base_chains.py` | Table 12, Appendix I |
| `quality_filtered_rerun.py`, `quality_filtered_figure5.py` → `quality_filtered_numbers.py` | Appendix J |
| `balanced_aggregation.py` | Section 3.2 |
| `token_counts_by_level.py` | Table 17, Appendix N |

Arrows mark scripts that read the output of the previous one;
`euclidean_deflation_sweep.py` imports `two_orderings.py`.

**4. Figures.** `python analysis/paper_figures/make_paper_figures.py`.

## Conventions

- The paper's *control* arm is called `commonality` in the data and code.
- `analysis/utils.py` numbers the seven pair categories in construction order;
  the paper swaps the first two (paper C1 = exact translation = code C2).
  `analysis/robustness/_common.py` converts between the two.
- The core-and-spread ordering called **B** in the paper is stored as `S`
  (`frac_S`, `frac_S_plus`) in the two-orderings outputs.
- Seeds are fixed (`SEED = 42`).

## Citation

```bibtex
@misc{godoyortiz2026meaning,
  title  = {Meaning Contracts: Local Semantic Geometry in Large Language Models},
  author = {Godoy Ortiz, Andr{\'e}s Daniel},
  year   = {2026},
  note   = {Preprint}
}
```
