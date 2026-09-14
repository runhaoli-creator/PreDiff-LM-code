"""
Evaluation utilities for Masked Diffusion LM.

Key metrics:
  1. ELBO-based NLL/PPL (proper upper bound on negative log-likelihood)
  2. Token accuracy at various mask rates
  3. Generation quality via sampler + standard metrics (BLEU, distinct-n, etc.)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from samplers.masked_diffusion_sampler import MaskedDiffusionSampler, SamplerConfig


@torch.no_grad()
def compute_elbo_ppl(
    model: nn.Module,
    loader: DataLoader,
    n_time_steps: int = 64,
    max_batches: int = 100,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute ELBO-based perplexity for masked diffusion model.

    Uses numerical integration (midpoint rule) over mask rates:
        NLL ≈ (1/T) Σ_{k=0}^{T-1} [1/(1-t_k)] * E[CE at mask_rate=t_k]

    where t_k = (k + 0.5) / T.

    Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (2024)

    Args:
        model:        MaskedDiffusionLM (already on device, eval mode).
        loader:       DataLoader yielding {prompt_ids, target_ids}.
        n_time_steps: Number of time steps for numerical integration.
        max_batches:  Maximum batches to evaluate.
        device:       Target device.

    Returns:
        {"nll": float, "ppl": float, "details": dict}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_weighted_ce = 0.0
    total_tokens = 0
    per_t_ce = {}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        B, L_y = target_ids.shape

        for t_idx in range(n_time_steps):
            t_val = (t_idx + 0.5) / n_time_steps
            mask_rate = torch.full((B,), t_val, device=device)
            mask = torch.rand(B, L_y, device=device) < t_val

            # Ensure at least one masked
            no_mask = mask.sum(dim=1) == 0
            if no_mask.any():
                idx = torch.randint(L_y, (no_mask.sum(),), device=device)
                mask[no_mask, idx] = True

            out = model(
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                mode="train",
                mask=mask,
                mask_rate=mask_rate,
            )

            ce = out["ce_loss"].item()
            weight = (1.0 / max(1.0 - t_val, 1e-5)) * (1.0 / n_time_steps)
            total_weighted_ce += ce * weight * B * L_y

            # Track per-t CE for analysis
            per_t_ce.setdefault(t_idx, []).append(ce)

        total_tokens += B * L_y

    avg_nll = total_weighted_ce / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 100.0))

    return {
        "nll": avg_nll,
        "ppl": ppl,
        "details": {
            f"t={((k + 0.5) / n_time_steps):.2f}": sum(v) / len(v)
            for k, v in per_t_ce.items()
        },
    }


@torch.no_grad()
def evaluate_accuracy_by_mask_rate(
    model: nn.Module,
    loader: DataLoader,
    mask_rates: list[float] = None,
    max_batches: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """Evaluate masked token accuracy at different mask rates.

    Args:
        model:      MaskedDiffusionLM.
        loader:     DataLoader.
        mask_rates: List of mask rates to evaluate (default: 0.1 to 0.9).
        max_batches: Maximum batches.
        device:     Target device.

    Returns:
        {mask_rate: {"accuracy": float, "ce_loss": float}}
    """
    if mask_rates is None:
        mask_rates = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = {}

    for rate in mask_rates:
        total_acc = 0.0
        total_ce = 0.0
        n = 0

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            prompt_ids = batch["prompt_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            B, L_y = target_ids.shape

            mask_rate = torch.full((B,), rate, device=device)
            mask = torch.rand(B, L_y, device=device) < rate
            no_mask = mask.sum(dim=1) == 0
            if no_mask.any():
                idx = torch.randint(L_y, (no_mask.sum(),), device=device)
                mask[no_mask, idx] = True

            out = model(
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                mode="train",
                mask=mask,
                mask_rate=mask_rate,
            )

            total_acc += out["masked_acc"].item()
            total_ce += out["ce_loss"].item()
            n += 1

        results[rate] = {
            "accuracy": total_acc / max(n, 1),
            "ce_loss": total_ce / max(n, 1),
        }

    return results


@torch.no_grad()
def generate_and_evaluate(
    model: nn.Module,
    loader: DataLoader,
    sampler_config: dict = None,
    tokenizer_name: str = "gpt2",
    n_samples: int = 200,
    device: Optional[torch.device] = None,
) -> dict:
    """Generate text completions and compute quality metrics.

    Args:
        model:          MaskedDiffusionLM.
        loader:         DataLoader yielding {prompt_ids, target_ids}.
        sampler_config: Dict for SamplerConfig (default: 16 steps, confidence).
        tokenizer_name: HF tokenizer for decoding.
        n_samples:      Number of samples to generate.
        device:         Target device.

    Returns:
        {"bleu": float, "rouge1": float, "rougeL": float,
         "distinct1": float, "distinct2": float, "distinct3": float,
         "token_accuracy": float,
         "samples": list of dicts with prompt/reference/generated}
    """
    from eval.generation import compute_bleu, compute_rouge, distinct_n

    if device is None:
        device = next(model.parameters()).device
    if sampler_config is None:
        sampler_config = {"n_steps": 16, "strategy": "confidence", "schedule": "cosine"}

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sampler = MaskedDiffusionSampler(SamplerConfig(**sampler_config))

    references = []
    hypotheses = []
    token_accs = []
    sample_details = []

    generated = 0
    for batch in loader:
        if generated >= n_samples:
            break

        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        B = prompt_ids.shape[0]
        L_y = target_ids.shape[1]

        # Generate
        result = sampler.sample(model, prompt_ids, target_len=L_y)
        gen_ids = result["token_ids"]

        # Token accuracy
        match = (gen_ids == target_ids).float().mean().item()
        token_accs.append(match)

        # Decode
        for b in range(min(B, n_samples - generated)):
            ref_text = tokenizer.decode(target_ids[b].cpu().tolist(), skip_special_tokens=True)
            gen_text = tokenizer.decode(gen_ids[b].cpu().tolist(), skip_special_tokens=True)
            prompt_text = tokenizer.decode(prompt_ids[b].cpu().tolist(), skip_special_tokens=True)

            references.append(ref_text)
            hypotheses.append(gen_text)

            if len(sample_details) < 10:
                sample_details.append({
                    "prompt": prompt_text[:200],
                    "reference": ref_text[:200],
                    "generated": gen_text[:200],
                })

            generated += 1

    # Compute metrics
    bleu = compute_bleu(references, hypotheses) if references else {"bleu": 0.0}
    rouge = compute_rouge(references, hypotheses) if references else {"rouge1": 0.0, "rougeL": 0.0}

    gen_tokens = [tokenizer.encode(h) for h in hypotheses]
    d1 = distinct_n(gen_tokens, 1) if gen_tokens else 0.0
    d2 = distinct_n(gen_tokens, 2) if gen_tokens else 0.0
    d3 = distinct_n(gen_tokens, 3) if gen_tokens else 0.0

    return {
        "bleu": bleu.get("bleu", 0.0),
        "rouge1": rouge.get("rouge1", 0.0),
        "rougeL": rouge.get("rougeL", 0.0),
        "distinct1": d1,
        "distinct2": d2,
        "distinct3": d3,
        "token_accuracy": sum(token_accs) / len(token_accs) if token_accs else 0.0,
        "samples": sample_details,
    }
