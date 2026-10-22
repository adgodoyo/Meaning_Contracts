#!/usr/bin/env python3
"""
extract_activations.py — Extract hidden states from a language model on local hardware.

Extracts both last-token and mean-pooled representations at every layer
for the multilingual and constraints datasets.

Usage:
    python extraction/extract_activations.py --model gemma4-e4b
    python extraction/extract_activations.py --model gemma4-e4b --batch-size 2 --dtype float32

Outputs (in results/activations/{model_key}/):
    {model_key}_multilingual_last.npy   (N_ml, n_layers, hidden_dim)
    {model_key}_multilingual_mean.npy   (N_ml, n_layers, hidden_dim)
    {model_key}_constraints_last.npy    (N_ct, n_layers, hidden_dim)
    {model_key}_constraints_mean.npy    (N_ct, n_layers, hidden_dim)
    {model_key}_metadata.json           model info, sentence order, normalised depth
    {model_key}_extraction.log          full log

Resilience:
    Checkpoints after every batch. Resumes automatically on restart.
    Per-sentence error handling — skips and logs failures, never crashes.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "activations"

# ── Model registry ───────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "pythia-2.8b": {
        "hf_id": "EleutherAI/pythia-2.8b",
        "default_dtype": "bfloat16",
        "notes": "GPT-NeoX architecture",
    },
    "llama3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B",
        "default_dtype": "bfloat16",
        "notes": "Llama 3.2 3B base; gated, needs a Hugging Face token",
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B",
        "default_dtype": "bfloat16",
        "notes": "Qwen 2.5 3B base",
    },
    "gemma4-e4b": {
        "hf_id": "google/gemma-4-E4B",
        "default_dtype": "bfloat16",
        "notes": "Gemma 4 E4B; per-layer embeddings, config nested under text_config",
    },
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_multilingual(path):
    """Load multilingual JSONL → list of sentence dicts."""
    sentences = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for s in record["sentences"]:
                sentences.append({
                    "text": s["text"],
                    "type": s["type"],
                    "factor_id": record["factor_id"],
                    "language": record["language"],
                    "domain": record["domain"],
                    "adjective": record["adjective"],
                })
    return sentences


def load_constraints(path):
    """Load constraints flat JSONL → list of sentence dicts."""
    sentences = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            sentences.append({
                "text": row["sentence"],
                "chain_id": row["chain_id"],
                "arm": row["arm"],
                "register": row["register"],
                "level": row["level"],
                "concept": row["concept"],
            })
    return sentences


# ── Device selection ─────────────────────────────────────────────────────────

def get_device(force=None):
    """Select best available device. GPU unless CPU is explicitly forced."""
    if force == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Extraction engine ────────────────────────────────────────────────────────

def extract_hidden_states(
    sentences,
    model,
    tokenizer,
    n_layers,
    hidden_dim,
    device,
    output_prefix,
    batch_size,
    logger,
):
    """
    Extract hidden states for a list of sentences.
    Saves both last-token and mean-pooled representations.
    Checkpoints after every batch for crash resilience.

    Saves:
        {output_prefix}_last.npy  — shape (N, n_layers, hidden_dim)
        {output_prefix}_mean.npy  — shape (N, n_layers, hidden_dim)
    """
    N = len(sentences)
    checkpoint_path = f"{output_prefix}_checkpoint.json"
    last_path = f"{output_prefix}_last.npy"
    mean_path = f"{output_prefix}_mean.npy"

    # ── Resume from checkpoint ──
    start_idx = 0
    skipped = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt["last_completed_index"] + 1
        skipped = ckpt.get("skipped", [])
        logger.info(f"Resuming from sentence {start_idx}/{N} (skipped so far: {len(skipped)})")

    # ── Allocate or open memory-mapped arrays ──
    # Arrays live on disk, not in RAM — only the current batch slice is in memory.
    # open_memmap in 'r+' mode resumes into an existing file without zeroing it.
    shape = (N, n_layers, hidden_dim)
    size_mb = N * n_layers * hidden_dim * 4 / 1e6
    if start_idx == 0 or not os.path.exists(last_path):
        acts_last = np.lib.format.open_memmap(last_path, mode="w+", dtype=np.float32, shape=shape)
        acts_mean = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32, shape=shape)
        logger.info(f"Created memmap arrays: {shape}, {size_mb:.1f} MB each on disk")
    else:
        acts_last = np.lib.format.open_memmap(last_path, mode="r+")
        acts_mean = np.lib.format.open_memmap(mean_path, mode="r+")
        logger.info(f"Opened existing memmap arrays: shape={acts_last.shape}")

    # ── Extract ──
    t0 = time.time()

    for batch_start in range(start_idx, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_texts = [sentences[i]["text"] for i in range(batch_start, batch_end)]
        batch_num = (batch_start - start_idx) // batch_size

        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of tensors, each (B, T, H)
            attention_mask = inputs["attention_mask"]  # (B, T)
            B = attention_mask.shape[0]

            for layer_idx, hs in enumerate(hidden_states):
                if layer_idx >= n_layers:
                    break
                hs_f = hs.float()  # (B, T, H)

                # Last-token pooling
                lengths = attention_mask.sum(dim=1) - 1  # (B,)
                last_token = hs_f[torch.arange(B, device=device), lengths]  # (B, H)
                acts_last[batch_start:batch_end, layer_idx] = last_token.cpu().numpy()

                # Mean pooling
                mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
                summed = (hs_f * mask).sum(dim=1)  # (B, H)
                counts = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
                mean_pooled = summed / counts  # (B, H)
                acts_mean[batch_start:batch_end, layer_idx] = mean_pooled.cpu().numpy()

            # Free GPU/MPS memory after every batch
            del outputs, hidden_states, inputs, hs, hs_f, last_token, mean_pooled, summed, mask, counts
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
            for i in range(batch_start, batch_end):
                skipped.append(i)
                logger.warning(f"  Skipped sentence {i}: {sentences[i]['text'][:60]}")
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # ── Flush memmap slice to disk + checkpoint ──
        # Only the modified slice is written — not the whole array.
        acts_last.flush()
        acts_mean.flush()
        with open(checkpoint_path, "w") as f:
            json.dump({
                "last_completed_index": batch_end - 1,
                "total_sentences": N,
                "skipped": skipped,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f)

        # ── Progress ──
        elapsed = time.time() - t0
        done = batch_end - start_idx
        total = N - start_idx
        eta = (elapsed / done) * (total - done) if done > 0 else 0
        if (batch_num + 1) % 5 == 0 or batch_end == N:
            logger.info(
                f"  {batch_end:5d}/{N} ({100 * batch_end / N:.1f}%)  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  skipped={len(skipped)}"
            )

    # ── Close memmaps (data already on disk) ──
    del acts_last, acts_mean

    # Remove checkpoint (extraction complete)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(last_path) / 1e6
    logger.info(f"Done. {N} sentences in {elapsed:.1f}s. Skipped: {len(skipped)}")
    logger.info(f"  Last-token: {last_path} ({size_mb:.1f} MB)")
    logger.info(f"  Mean-pool:  {mean_path} ({size_mb:.1f} MB)")

    return skipped


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from a language model.")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model key from the registry")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Sentences per forward pass (default: 1 for MPS safety)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None,
                        help="Override model dtype (default: from registry)")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None,
                        help="Force device (default: auto-detect)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # ── Config ──
    model_key = args.model
    config = MODEL_CONFIGS[model_key]
    hf_id = config["hf_id"]
    dtype_str = args.dtype or config["default_dtype"]
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype_str]
    device = get_device(args.device)

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──
    log_path = output_dir / f"{model_key}_extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"MODEL:   {model_key} ({hf_id})")
    logger.info(f"DEVICE:  {device}")
    logger.info(f"DTYPE:   {dtype_str}")
    logger.info(f"BATCH:   {args.batch_size}")
    logger.info(f"DATA:    {data_dir}")
    logger.info(f"OUTPUT:  {output_dir}")
    logger.info(f"NOTES:   {config['notes']}")
    logger.info("=" * 60)

    # ── Load datasets ──
    ml_path = data_dir / "multilingual" / "multilingual.jsonl"
    ct_path = data_dir / "constraints" / "constraints_sentences.jsonl"

    logger.info(f"Loading multilingual from {ml_path}")
    ml_sentences = load_multilingual(ml_path)
    logger.info(f"  → {len(ml_sentences)} sentences")

    logger.info(f"Loading constraints from {ct_path}")
    ct_sentences = load_constraints(ct_path)
    logger.info(f"  → {len(ct_sentences)} sentences")

    # ── Load model ──
    logger.info(f"Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {hf_id} (dtype={dtype_str}, this may take a minute...)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        token=args.hf_token,
    )
    model.eval()
    model.to(device)

    # Some architectures (e.g. Gemma 4 PLE) nest config under text_config.
    # Fall back to None — the probe step below will resolve the real n_layers.
    def _get_config_attr(cfg, attr):
        """Try top-level config, then text_config, then return None."""
        val = getattr(cfg, attr, None)
        if val is None and hasattr(cfg, "text_config"):
            val = getattr(cfg.text_config, attr, None)
        return val

    _n_hidden = _get_config_attr(model.config, "num_hidden_layers")
    n_layers = (_n_hidden + 1) if _n_hidden is not None else None  # +1 for embedding
    _hidden = _get_config_attr(model.config, "hidden_size")
    hidden_dim = _hidden  # may also be None — resolved from probe below
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    logger.info(f"Model loaded in {time.time() - t0:.1f}s")
    logger.info(f"  Parameters: {n_params:.2f}B")
    if n_layers is not None:
        logger.info(f"  Layers (from config): {n_layers} (1 embedding + {n_layers - 1} transformer)")
    else:
        logger.info("  Layers (from config): unknown — will be determined by probe")
    if hidden_dim is not None:
        logger.info(f"  Hidden dim: {hidden_dim}")

    # ── Probe: verify hidden_states output on first sentence ──
    # This resolves n_layers / hidden_dim for architectures whose config is nested
    # (e.g. Gemma 4 PLE which puts these under config.text_config).
    logger.info("Probing model with first sentence to verify output_hidden_states...")
    test_input = tokenizer(ml_sentences[0]["text"], return_tensors="pt").to(device)
    with torch.no_grad():
        test_out = model(**test_input, output_hidden_states=True)
    hs = test_out.hidden_states
    probe_n_layers = len(hs)
    probe_hidden_dim = hs[0].shape[-1]
    logger.info(f"  output_hidden_states: {probe_n_layers} tensors, first shape={hs[0].shape}")
    if n_layers is None:
        logger.info(f"  n_layers resolved from probe: {probe_n_layers}")
        n_layers = probe_n_layers
    elif probe_n_layers != n_layers:
        logger.warning(f"  EXPECTED {n_layers} layers but got {probe_n_layers}! Using probe value.")
        n_layers = probe_n_layers
    if hidden_dim is None:
        logger.info(f"  hidden_dim resolved from probe: {probe_hidden_dim}")
        hidden_dim = probe_hidden_dim
    elif probe_hidden_dim != hidden_dim:
        logger.warning(f"  Config hidden_dim={hidden_dim} but probe shows {probe_hidden_dim}! Using probe value.")
        hidden_dim = probe_hidden_dim
    logger.info(f"  Layers: {n_layers} | Hidden dim: {hidden_dim}")
    del test_out, test_input

    # ── Extract multilingual ──
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTING MULTILINGUAL ({len(ml_sentences)} sentences)")
    logger.info(f"{'='*60}")
    ml_prefix = str(output_dir / f"{model_key}_multilingual")
    ml_skipped = extract_hidden_states(
        ml_sentences, model, tokenizer, n_layers, hidden_dim,
        device, ml_prefix, args.batch_size, logger,
    )

    # ── Extract constraints ──
    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTING CONSTRAINTS ({len(ct_sentences)} sentences)")
    logger.info(f"{'='*60}")
    ct_prefix = str(output_dir / f"{model_key}_constraints")
    ct_skipped = extract_hidden_states(
        ct_sentences, model, tokenizer, n_layers, hidden_dim,
        device, ct_prefix, args.batch_size, logger,
    )

    # ── Save metadata ──
    normalised_depth = [round(i / (n_layers - 1), 4) for i in range(n_layers)]

    metadata = {
        "model_key": model_key,
        "hf_id": hf_id,
        "num_layers": n_layers,
        "hidden_dim": hidden_dim,
        "parameters_B": round(n_params, 3),
        "dtype": dtype_str,
        "device": str(device),
        "batch_size": args.batch_size,
        "pooling_methods": ["last_token", "mean"],
        "normalised_depth": normalised_depth,
        "multilingual": {
            "num_sentences": len(ml_sentences),
            "skipped": ml_skipped,
            "factor_ids": [s["factor_id"] for s in ml_sentences],
            "languages": [s["language"] for s in ml_sentences],
            "types": [s["type"] for s in ml_sentences],
        },
        "constraints": {
            "num_sentences": len(ct_sentences),
            "skipped": ct_skipped,
            "chain_ids": [s["chain_id"] for s in ct_sentences],
            "arms": [s["arm"] for s in ct_sentences],
            "levels": [s["level"] for s in ct_sentences],
        },
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = output_dir / f"{model_key}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")

    # ── Verify ──
    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION")
    logger.info(f"{'='*60}")

    for dataset_name in ["multilingual", "constraints"]:
        for pooling in ["last", "mean"]:
            path = output_dir / f"{model_key}_{dataset_name}_{pooling}.npy"
            acts = np.load(path)
            n_nan = int(np.isnan(acts).sum())
            n_inf = int(np.isinf(acts).sum())
            status = "PASS" if n_nan == 0 and n_inf == 0 else "FAIL"
            logger.info(f"  {dataset_name}/{pooling}: shape={acts.shape}  "
                        f"NaN={n_nan} Inf={n_inf}  {status}")

    # ── Cleanup ──
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"\nAll outputs in: {output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
