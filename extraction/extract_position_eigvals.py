"""
extraction/extract_position_eigvals.py

Extract per-token-position eigenvalues for the position analysis script.

For each model, tokenizes sentences from constraints_sentences.jsonl, runs a
forward pass capturing all layer hidden states, then groups sentences by
(chain_id, arm, level).  For each group at each (layer, token_position) bin,
it computes a covariance and extracts the top-15 eigenvalues via truncated SVD.

Output: results/caches/position/mean/{model}_eigvals.npz
Keys:
  ref_eigvals    (n_ref_chains, n_layers, 11, n_levels, 15)  float64
  com_eigvals    (n_com_chains, n_layers, 11, n_levels, 15)  float64
  ref_chain_ids  (n_ref_chains,)  str
  com_chain_ids  (n_com_chains,)  str
  positions      (11,)  float64  — [0.0, 0.1, ..., 1.0]
  n_layers       int
  hidden_dim     int

Usage:
  python extraction/extract_position_eigvals.py --model gemma4-e4b
  python extraction/extract_position_eigvals.py --model all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import numpy as np
from sklearn.utils.extmath import randomized_svd

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from utils import MODELS, get_cache_dir

SEED = 42

MODEL_CONFIGS = {
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b",   "dtype": "bfloat16"},
    "llama3.2-3b": {"hf_id": "meta-llama/Llama-3.2-3B",   "dtype": "bfloat16"},
    "qwen2.5-3b":  {"hf_id": "Qwen/Qwen2.5-3B",            "dtype": "bfloat16"},
    "gemma4-e4b":  {"hf_id": "google/gemma-4-E4B",          "dtype": "bfloat16"},
}

N_POSITIONS  = 11                   # bins: 0.0, 0.1, ..., 1.0
POSITIONS    = np.linspace(0, 1, N_POSITIONS)
N_TOP_EIGVALS = 15
N_LEVELS     = 5                    # S0..S4
ARM_REF      = "refinement"
ARM_COM      = "commonality"

_ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH   = _ROOT / "data" / "constraints" / "constraints_sentences.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sentences() -> dict[tuple[str, str, int], list[str]]:
    """
    Returns a dict mapping (chain_id, arm, level) → list of sentences.
    Only keeps ARM_REF and ARM_COM.
    """
    groups: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    with open(DATA_PATH) as f:
        for line in f:
            rec = json.loads(line)
            arm = rec["arm"]
            if arm not in (ARM_REF, ARM_COM):
                continue
            key = (rec["chain_id"], arm, int(rec["level"]))
            groups[key].append(rec["sentence"])
    return dict(groups)


# ---------------------------------------------------------------------------
# Eigenvalue extraction helpers
# ---------------------------------------------------------------------------

def position_bin(token_idx: int, seq_len: int) -> int:
    """Map token index to one of 11 position bins (0..10)."""
    if seq_len <= 1:
        return 0
    p = token_idx / (seq_len - 1)
    return int(round(p * (N_POSITIONS - 1)))


def eigvals_from_matrix(X: np.ndarray, n_components: int = N_TOP_EIGVALS) -> np.ndarray:
    """
    X: (N, D) float64.  Returns top-n eigenvalues of the covariance, float64.
    Falls back to fewer components if N is too small.
    """
    if X.shape[0] < 2:
        return np.zeros(n_components, dtype=np.float64)
    X64 = X.astype(np.float64)
    mu  = X64.mean(axis=0)
    Xc  = X64 - mu
    n_comp = min(n_components, min(Xc.shape) - 1)
    if n_comp < 1:
        return np.zeros(n_components, dtype=np.float64)
    np.random.seed(SEED)
    _, S, _ = randomized_svd(Xc, n_components=n_comp, random_state=SEED)
    # Convert singular values of centred data to eigenvalues of covariance
    N = X64.shape[0]
    eigvals = (S ** 2) / max(N - 1, 1)
    # Pad if we got fewer than n_components
    if len(eigvals) < n_components:
        eigvals = np.concatenate([eigvals, np.zeros(n_components - len(eigvals))])
    return eigvals.astype(np.float64)


# ---------------------------------------------------------------------------
# Hidden-state extraction with hooks
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    sentences: list[str],
    device: "torch.device",
    batch_size: int = 8,
) -> list[np.ndarray]:
    """
    Returns list of arrays, one per sentence: shape (n_layers, seq_len, hidden_dim).
    n_layers = number of transformer layers + 1 (embedding layer at index 0).
    """
    results = []

    for start in range(0, len(sentences), batch_size):
        batch_sents = sentences[start : start + batch_size]

        enc = tokenizer(
            batch_sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # out.hidden_states: tuple of (batch, seq_len, hidden_dim), len = n_layers+1
        # Stack to (n_layers+1, batch, seq_len, hidden_dim) then rearrange
        hidden = torch.stack(out.hidden_states, dim=0)  # (L, B, T, D)
        hidden = hidden.permute(1, 0, 2, 3).cpu().float().numpy()  # (B, L, T, D)

        masks = attention_mask.cpu().numpy()  # (B, T)

        for b_idx in range(len(batch_sents)):
            seq_len = int(masks[b_idx].sum())
            # Trim to actual sequence length
            hs = hidden[b_idx, :, :seq_len, :]  # (L, seq_len, D)
            results.append(hs)

        del enc, input_ids, attention_mask, out, hidden, masks
        if _TORCH_AVAILABLE:
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Per-group eigenvalue computation
# ---------------------------------------------------------------------------

def compute_group_eigvals(
    hidden_states_list: list[np.ndarray],
) -> np.ndarray:
    """
    hidden_states_list: list of (n_layers, seq_len, hidden_dim) arrays.
    Returns: (n_layers, N_POSITIONS, N_TOP_EIGVALS) float64.

    For each (layer, position bin): collect hidden states from all sentences
    that have a token mapping to that bin, then compute covariance eigenvalues.
    """
    if not hidden_states_list:
        return np.zeros((0, N_POSITIONS, N_TOP_EIGVALS), dtype=np.float64)

    n_layers  = hidden_states_list[0].shape[0]
    result    = np.zeros((n_layers, N_POSITIONS, N_TOP_EIGVALS), dtype=np.float64)

    # Collect per-position hidden states for each layer
    for layer_idx in range(n_layers):
        pos_buckets: list[list[np.ndarray]] = [[] for _ in range(N_POSITIONS)]

        for hs in hidden_states_list:
            seq_len = hs.shape[1]
            for tok_idx in range(seq_len):
                p_bin = position_bin(tok_idx, seq_len)
                pos_buckets[p_bin].append(hs[layer_idx, tok_idx, :])

        for p_bin in range(N_POSITIONS):
            bucket = pos_buckets[p_bin]
            if len(bucket) < 2:
                continue
            X = np.stack(bucket, axis=0)  # (N, D)
            result[layer_idx, p_bin, :] = eigvals_from_matrix(X)

    return result


# ---------------------------------------------------------------------------
# Main per-model extraction
# ---------------------------------------------------------------------------

def extract_model(model_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"[{model_name}] Loading model...")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    cfg    = MODEL_CONFIGS[model_name]
    hf_id  = cfg["hf_id"]
    dtype  = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float32

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for extraction.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Determine number of layers from a dummy forward pass
    dummy_ids = torch.zeros((1, 4), dtype=torch.long).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_ids, output_hidden_states=True)
    n_layers   = len(dummy_out.hidden_states)    # includes embedding layer
    hidden_dim = dummy_out.hidden_states[0].shape[-1]
    del dummy_ids, dummy_out
    print(f"  n_layers={n_layers}  hidden_dim={hidden_dim}")

    # Load sentence groups
    all_groups = load_sentences()
    print(f"  Loaded {len(all_groups)} (chain, arm, level) groups from {DATA_PATH}")

    # Separate ref and com chains, sorted for reproducibility
    ref_chains = sorted({k[0] for k in all_groups if k[1] == ARM_REF})
    com_chains = sorted({k[0] for k in all_groups if k[1] == ARM_COM})
    print(f"  ref_chains={len(ref_chains)}  com_chains={len(com_chains)}")

    # Output path
    out_dir = get_cache_dir("position", "mean")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_eigvals.npz"

    # Load checkpoint if it exists
    checkpoint_path = out_dir / f"{model_name}_eigvals_checkpoint.npz"
    ref_eigvals  = np.zeros((len(ref_chains), n_layers, N_POSITIONS, N_LEVELS, N_TOP_EIGVALS), dtype=np.float64)
    com_eigvals  = np.zeros((len(com_chains), n_layers, N_POSITIONS, N_LEVELS, N_TOP_EIGVALS), dtype=np.float64)
    ref_done     = np.zeros(len(ref_chains), dtype=bool)
    com_done     = np.zeros(len(com_chains), dtype=bool)

    if checkpoint_path.exists():
        print(f"  [checkpoint] loading from {checkpoint_path}")
        ckpt = np.load(str(checkpoint_path), allow_pickle=True)
        if "ref_eigvals" in ckpt:
            ref_eigvals[:] = ckpt["ref_eigvals"]
        if "com_eigvals" in ckpt:
            com_eigvals[:] = ckpt["com_eigvals"]
        if "ref_done" in ckpt:
            ref_done[:] = ckpt["ref_done"]
        if "com_done" in ckpt:
            com_done[:] = ckpt["com_done"]
        del ckpt

    def _save_checkpoint() -> None:
        np.savez(
            str(checkpoint_path),
            ref_eigvals=ref_eigvals,
            com_eigvals=com_eigvals,
            ref_done=ref_done,
            com_done=com_done,
        )

    # --- Process refinement chains ---
    for chain_idx, chain_id in enumerate(ref_chains):
        if ref_done[chain_idx]:
            print(f"  [ref {chain_idx+1}/{len(ref_chains)}] {chain_id} — already done, skipping")
            continue

        print(f"  [ref {chain_idx+1}/{len(ref_chains)}] {chain_id}")

        for level in range(N_LEVELS):
            key = (chain_id, ARM_REF, level)
            sentences = all_groups.get(key, [])
            if not sentences:
                print(f"    WARNING: no sentences for {key}")
                continue

            hidden_states_list = extract_hidden_states(model, tokenizer, sentences, device)

            # (n_layers, N_POSITIONS, N_TOP_EIGVALS)
            group_ev = compute_group_eigvals(hidden_states_list)

            ref_eigvals[chain_idx, :, :, level, :] = group_ev

        ref_done[chain_idx] = True
        _save_checkpoint()
        print(f"    checkpoint saved ({chain_idx+1}/{len(ref_chains)} ref chains done)")

    # --- Process commonality chains ---
    for chain_idx, chain_id in enumerate(com_chains):
        if com_done[chain_idx]:
            print(f"  [com {chain_idx+1}/{len(com_chains)}] {chain_id} — already done, skipping")
            continue

        print(f"  [com {chain_idx+1}/{len(com_chains)}] {chain_id}")

        for level in range(N_LEVELS):
            key = (chain_id, ARM_COM, level)
            sentences = all_groups.get(key, [])
            if not sentences:
                print(f"    WARNING: no sentences for {key}")
                continue

            hidden_states_list = extract_hidden_states(model, tokenizer, sentences, device)
            group_ev = compute_group_eigvals(hidden_states_list)
            com_eigvals[chain_idx, :, :, level, :] = group_ev

        com_done[chain_idx] = True
        _save_checkpoint()
        print(f"    checkpoint saved ({chain_idx+1}/{len(com_chains)} com chains done)")

    # --- Save final output ---
    # Reshape: current (n_chains, n_layers, N_POSITIONS, N_LEVELS, N_TOP_EIGVALS)
    # Target:  (n_chains, n_layers, N_POSITIONS, N_LEVELS, N_TOP_EIGVALS) — same
    np.savez(
        str(out_path),
        ref_eigvals=ref_eigvals,
        com_eigvals=com_eigvals,
        ref_chain_ids=np.array(ref_chains),
        com_chain_ids=np.array(com_chains),
        positions=POSITIONS,
        n_layers=np.int64(n_layers),
        hidden_dim=np.int64(hidden_dim),
    )
    print(f"\n  saved → {out_path}")
    print(f"  ref_eigvals shape: {ref_eigvals.shape}")
    print(f"  com_eigvals shape: {com_eigvals.shape}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  checkpoint deleted")

    # Free GPU memory
    del model
    if _TORCH_AVAILABLE:
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-token-position eigenvalues for position analysis."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f"Model name or 'all'. Choices: {list(MODEL_CONFIGS.keys()) + ['all']}",
    )
    args = parser.parse_args()

    if args.model == "all":
        models_to_run = MODELS
    elif args.model in MODEL_CONFIGS:
        models_to_run = [args.model]
    else:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODEL_CONFIGS.keys()) + ['all']}")
        sys.exit(1)

    for model_name in models_to_run:
        extract_model(model_name)


if __name__ == "__main__":
    main()
