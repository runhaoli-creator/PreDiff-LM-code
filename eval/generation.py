"""
Generation quality metrics for CFlow-LM.

Covers:
  - Distinct-n (diversity)
  - Repetition rate
  - MAUVE score (distribution similarity)
  - BLEU / ROUGE (overlap-based)
  - Code exact match / prefix match
  - MAUVE requires running a reference model; we use a light version.
"""

from __future__ import annotations

import collections
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Diversity & repetition
# ──────────────────────────────────────────────────────────────────────────────

def distinct_n(token_lists: list[list[int]], n: int) -> float:
    """Compute Distinct-n: fraction of unique n-grams across all sequences.

    Args:
        token_lists: List of token sequences (each is a list of int token ids).
        n: N-gram order.
    Returns:
        Fraction of unique n-grams in [0, 1].
    """
    all_ngrams: set = set()
    total = 0
    for seq in token_lists:
        ngrams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
        all_ngrams.update(ngrams)
        total += len(ngrams)
    if total == 0:
        return 0.0
    return len(all_ngrams) / total


def repetition_rate(token_lists: list[list[int]], window: int = 16) -> float:
    """Fraction of tokens that appear in a recent context window.

    A simple proxy for repetition: for each token, check how many of the
    previous `window` tokens are identical to it.

    Args:
        token_lists: List of token sequences.
        window:      Context window size.
    Returns:
        Mean repetition rate across sequences.
    """
    rates = []
    for seq in token_lists:
        if len(seq) <= window:
            continue
        rep = sum(
            seq[i] in seq[max(0, i - window) : i]
            for i in range(window, len(seq))
        )
        rates.append(rep / (len(seq) - window))
    return sum(rates) / len(rates) if rates else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# BLEU / ROUGE
# ──────────────────────────────────────────────────────────────────────────────

def compute_bleu(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus BLEU using sacrebleu.

    Returns dict with "bleu" (float 0-100) and "bp" (brevity penalty).
    """
    try:
        import sacrebleu
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return {"bleu": result.score, "bp": result.bp}
    except ImportError:
        return {"bleu": float("nan"), "bp": float("nan")}


def compute_rouge(references: list[str], hypotheses: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L using the evaluate library."""
    try:
        import evaluate as hf_eval
        rouge = hf_eval.load("rouge")
        result = rouge.compute(predictions=hypotheses, references=references)
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
        }
    except ImportError:
        return {"rouge1": float("nan"), "rouge2": float("nan"), "rougeL": float("nan")}


# ──────────────────────────────────────────────────────────────────────────────
# MAUVE
# ──────────────────────────────────────────────────────────────────────────────

def compute_mauve(
    references: list[str],
    hypotheses: list[str],
    featurize_model_name: str = "gpt2",
    device_id: int = 0,
    max_len: int = 256,
) -> float:
    """Compute MAUVE score using the mauve-text library.

    MAUVE measures the information-theoretic divergence between human text
    and model text distributions via a maximum-mean discrepancy.

    Returns MAUVE score in [0, 1] (higher is better).
    """
    try:
        import mauve
        result = mauve.compute_mauve(
            p_text=references,
            q_text=hypotheses,
            device_id=device_id,
            max_text_length=max_len,
            verbose=False,
            featurize_model_name=featurize_model_name,
        )
        return float(result.mauve)
    except Exception as e:
        print(f"[MAUVE] Could not compute: {e}")
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Code metrics
# ──────────────────────────────────────────────────────────────────────────────

def code_exact_match(references: list[str], hypotheses: list[str]) -> float:
    """Exact match rate between reference and hypothesis code strings."""
    matches = sum(r.strip() == h.strip() for r, h in zip(references, hypotheses))
    return matches / len(references) if references else 0.0


def code_prefix_match(
    references: list[str],
    hypotheses: list[str],
    prefix_tokens: int = 50,
    tokenizer_name: str = "gpt2",
) -> float:
    """Fraction of tokens in the first prefix_tokens that match exactly.

    Used as a proxy for "does the model start the continuation correctly?"
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_ids = tokenizer.encode(ref)[:prefix_tokens]
        hyp_ids = tokenizer.encode(hyp)[:prefix_tokens]
        n = min(len(ref_ids), len(hyp_ids), prefix_tokens)
        if n == 0:
            continue
        matches = sum(r == h for r, h in zip(ref_ids[:n], hyp_ids[:n]))
        scores.append(matches / n)
    return sum(scores) / len(scores) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Generation wrapper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    sampler,
    loader: DataLoader,
    tokenizer,
    n_samples: int = 500,
    device: Optional[torch.device] = None,
) -> tuple[list[str], list[str]]:
    """Generate text continuations for a set of prompts.

    Args:
        model:     FlowDecoderLM.
        sampler:   ODESampler instance.
        loader:    DataLoader yielding {prompt_ids, target_ids}.
        tokenizer: HF tokenizer for decoding.
        n_samples: Number of samples to generate.
        device:    Target device.

    Returns:
        (references, hypotheses) – parallel lists of decoded strings.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    references, hypotheses = [], []

    for batch in tqdm(loader, desc="Generating samples"):
        if len(references) >= n_samples:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        B, L_y = target_ids.shape

        z0 = sampler.sample(model, prompt_ids, L_y)
        out = model(prompt_ids=prompt_ids, z_t=z0, t=torch.ones(B, device=device), mode="sample")
        pred_ids = out["token_ids"]  # (B, L_y)

        for b in range(B):
            ref = tokenizer.decode(target_ids[b].tolist(), skip_special_tokens=True)
            hyp = tokenizer.decode(pred_ids[b].tolist(), skip_special_tokens=True)
            references.append(ref)
            hypotheses.append(hyp)
            if len(references) >= n_samples:
                break

    return references, hypotheses
