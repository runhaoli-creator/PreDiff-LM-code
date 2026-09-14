"""
Perplexity evaluation for CFlow-LM and AR baselines.

Computes pseudo-perplexity for the flow decoder (via flow matching loss)
and standard next-token perplexity for AR models.

Note on CFlow-LM perplexity:
  True perplexity requires a tractable log-likelihood, which CNFs do not
  trivially provide. We report:
    1. "Flow loss" (MSE in embedding space) as a proxy.
    2. "Token accuracy" at t=1 after ODE decoding.
    3. "Pseudo perplexity" via vocab-projection cross-entropy at t→1.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_flow_loss(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int = 100,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute mean flow matching loss (proxy for perplexity) on a dataloader.

    Args:
        model:       FlowDecoderLM instance.
        loader:      DataLoader yielding {prompt_ids, target_ids}.
        max_batches: Cap on number of batches to evaluate.
        device:      Target device; inferred from model if None.

    Returns:
        dict with "flow_loss" (float) and "n_tokens" (int).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_tokens = 0

    for i, batch in enumerate(tqdm(loader, desc="Eval (flow loss)", leave=False)):
        if i >= max_batches:
            break
        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        out = model(prompt_ids=prompt_ids, target_ids=target_ids, mode="train")
        B, L = target_ids.shape
        total_loss += out["loss"].item()
        n_tokens += B * L
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"flow_loss": avg_loss, "n_tokens": n_tokens}


@torch.no_grad()
def compute_ar_perplexity(
    model: nn.Module,
    loader: DataLoader,
    max_batches: int = 100,
    device: Optional[torch.device] = None,
    stride: int = 512,
) -> dict:
    """Compute standard sliding-window perplexity for an AR language model.

    Supports GPT-2 from HuggingFace (or any CausalLM with .forward returning logits).

    Args:
        model:       HF CausalLM.
        loader:      DataLoader yielding {input_ids}.
        max_batches: Cap.
        device:      Target device.
        stride:      Sliding window stride for long sequences.

    Returns:
        dict with "perplexity" (float) and "nll" (float).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll = 0.0
    n_tokens = 0

    for i, batch in enumerate(tqdm(loader, desc="Eval (AR perplexity)", leave=False)):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        B, L = input_ids.shape

        # Slice into windows for long sequences
        for start in range(0, L - 1, stride):
            end = min(start + stride, L)
            chunk = input_ids[:, start:end]

            out = model(chunk, labels=chunk)
            # HF CausalLM returns loss = mean NLL over non-padding tokens
            chunk_tokens = (end - start - 1) * B
            total_nll += out.loss.item() * chunk_tokens
            n_tokens += chunk_tokens

    if n_tokens == 0:
        return {"perplexity": float("nan"), "nll": float("nan")}

    avg_nll = total_nll / n_tokens
    ppl = math.exp(avg_nll)
    return {"perplexity": ppl, "nll": avg_nll}


@torch.no_grad()
def compute_pseudo_perplexity(
    model: nn.Module,
    loader: DataLoader,
    sampler,
    max_batches: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute pseudo-perplexity by running the full ODE to get z_0 then
    measuring cross-entropy of the projected logits against the ground truth.

    This requires running the ODE sampler (expensive) so is used only at eval.

    Args:
        model:       FlowDecoderLM instance.
        loader:      DataLoader yielding {prompt_ids, target_ids}.
        sampler:     ODESampler instance.
        max_batches: Cap.
        device:      Target device.

    Returns:
        dict with "pseudo_ppl", "token_accuracy", "nll".
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0

    for i, batch in enumerate(tqdm(loader, desc="Eval (pseudo-PPL)", leave=False)):
        if i >= max_batches:
            break
        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        B, L_y = target_ids.shape

        # ODE sample
        z0 = sampler.sample(model, prompt_ids, L_y)  # (B, L_y, E)

        # Project to logits
        out = model(
            prompt_ids=prompt_ids,
            z_t=z0,
            t=torch.ones(B, device=device),
            mode="sample",
        )
        logits = out.get("logits")  # (B, L_y, V)
        if logits is None:
            continue

        # Cross-entropy
        V = logits.shape[-1]
        ce = F.cross_entropy(
            logits.view(B * L_y, V),
            target_ids.view(B * L_y),
            reduction="sum",
        )
        total_nll += ce.item()

        # Token accuracy
        preds = logits.argmax(dim=-1)  # (B, L_y)
        total_correct += (preds == target_ids).sum().item()
        total_tokens += B * L_y

    if total_tokens == 0:
        return {"pseudo_ppl": float("nan"), "token_accuracy": 0.0, "nll": float("nan")}

    avg_nll = total_nll / total_tokens
    pseudo_ppl = math.exp(min(avg_nll, 20))  # cap to avoid overflow
    token_acc = total_correct / total_tokens
    return {
        "pseudo_ppl": pseudo_ppl,
        "token_accuracy": token_acc,
        "nll": avg_nll,
    }
