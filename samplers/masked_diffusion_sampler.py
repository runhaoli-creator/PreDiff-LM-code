"""
Masked Diffusion Sampler: iterative unmasking for discrete diffusion generation.

Supports multiple unmasking strategies:
  - "confidence": unmask highest-confidence tokens first (Confidence-Adaptive Unmasking)
  - "random":     unmask random subset (standard baseline)
  - "entropy":    unmask lowest-entropy (most certain) tokens first

Supports multiple mask-rate schedules:
  - "linear":  mask_rate decreases linearly from 1 to 0
  - "cosine":  mask_rate follows cos(step/N * π/2) — slower near end
  - "sqrt":    mask_rate follows 1 - sqrt(step/N) — faster start

Key innovation: Confidence-Adaptive Unmasking (CAU) naturally handles
easy-to-predict tokens first (function words, determiners) and leaves
difficult tokens (content words, names) for later steps when more context
is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplerConfig:
    """Configuration for the masked diffusion sampler."""
    n_steps: int = 16
    strategy: str = "confidence"    # "confidence" | "random" | "entropy"
    schedule: str = "cosine"        # "linear" | "cosine" | "sqrt"
    temperature: float = 1.0       # sampling temperature
    top_p: float = 1.0             # nucleus sampling threshold
    use_self_cond: bool = True     # use self-conditioning during sampling


class MaskedDiffusionSampler:
    """Iterative unmasking sampler for MaskedDiffusionLM.

    Starting from all-masked target, progressively unmasks tokens
    over N steps using the specified strategy and schedule.

    Args:
        config: SamplerConfig with sampling hyperparameters.
    """

    def __init__(self, config: SamplerConfig):
        self.config = config

    @torch.no_grad()
    def sample(
        self,
        model,
        prompt_ids: torch.Tensor,
        target_len: int,
        initial_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """Generate target tokens by iterative unmasking.

        Args:
            model:       MaskedDiffusionLM instance.
            prompt_ids:  (B, Lx) tokenized prompt.
            target_len:  Number of target tokens to generate.
            initial_ids: (B, Ly) optional initial target IDs (e.g. for infill).

        Returns dict:
            "token_ids":  (B, Ly) generated token IDs
            "logits":     (B, Ly, V) final-step logits
            "confidence": (B, Ly) final-step confidence scores
            "steps_log":  list of per-step info dicts
        """
        cfg = self.config
        B = prompt_ids.shape[0]
        device = prompt_ids.device

        # Initialize: all positions masked
        if initial_ids is not None:
            target_ids = initial_ids.clone()
        else:
            target_ids = torch.zeros(B, target_len, dtype=torch.long, device=device)
        mask = torch.ones(B, target_len, dtype=torch.bool, device=device)

        x_self_cond = None
        steps_log = []

        for step in range(cfg.n_steps):
            # ── Compute target mask rate after this step ──────────────
            progress = (step + 1) / cfg.n_steps
            if cfg.schedule == "linear":
                target_mask_frac = 1.0 - progress
            elif cfg.schedule == "cosine":
                target_mask_frac = math.cos(progress * math.pi / 2)
            elif cfg.schedule == "sqrt":
                target_mask_frac = 1.0 - math.sqrt(progress)
            else:
                target_mask_frac = 1.0 - progress

            target_mask_frac = max(target_mask_frac, 0.0)
            target_n_masked = max(int(target_len * target_mask_frac), 0)

            # Current mask rate for model conditioning
            current_mask_rate = mask.float().mean(dim=1)  # (B,)

            # ── Forward pass ──────────────────────────────────────────
            out = model(
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                mode="sample",
                mask=mask,
                mask_rate=current_mask_rate,
                x_self_cond=x_self_cond if cfg.use_self_cond else None,
            )

            logits = out["logits"]  # (B, Ly, V)

            # Update self-conditioning
            if cfg.use_self_cond:
                x_self_cond = out.get("x_self_cond")

            # ── Sample tokens from logits ─────────────────────────────
            if cfg.temperature == 0 or (cfg.temperature == 1.0 and cfg.top_p >= 1.0):
                predicted = logits.argmax(dim=-1)  # (B, Ly)
            else:
                predicted = self._sample_tokens(logits)

            # Compute confidence for strategy selection
            probs = F.softmax(logits.float(), dim=-1)  # (B, Ly, V)
            confidence = probs.max(dim=-1).values       # (B, Ly)

            # ── Last step: unmask everything ──────────────────────────
            if step == cfg.n_steps - 1 or target_n_masked == 0:
                # Unmask all remaining
                new_tokens = target_ids.clone()
                new_tokens[mask] = predicted[mask]
                target_ids = new_tokens
                mask = torch.zeros_like(mask)

                steps_log.append({
                    "step": step,
                    "n_unmasked": target_len,
                    "mask_rate": 0.0,
                })
                break

            # ── Unmask tokens according to strategy ───────────────────
            n_currently_masked = mask.sum(dim=1)  # (B,)
            n_to_keep_masked = torch.full(
                (B,), target_n_masked, device=device, dtype=torch.long
            )
            n_to_unmask = (n_currently_masked - n_to_keep_masked).clamp(min=0)

            # Strategy: which masked tokens to unmask
            if cfg.strategy == "confidence":
                # Unmask highest-confidence masked tokens
                score = confidence.clone()
                score[~mask] = -1.0  # don't select already-unmasked
                self._unmask_topk(
                    target_ids, predicted, mask, score, n_to_unmask, high_is_good=True
                )
            elif cfg.strategy == "entropy":
                # Unmask lowest-entropy (most certain) masked tokens
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, Ly)
                score = entropy.clone()
                score[~mask] = float('inf')  # don't select already-unmasked
                self._unmask_topk(
                    target_ids, predicted, mask, score, n_to_unmask, high_is_good=False
                )
            else:  # random
                self._unmask_random(target_ids, predicted, mask, n_to_unmask)

            steps_log.append({
                "step": step,
                "n_unmasked": int(n_to_unmask.float().mean().item()),
                "mask_rate": float(mask.float().mean().item()),
                "avg_confidence": float(confidence[mask].mean().item()) if mask.any() else 1.0,
            })

        return {
            "token_ids": target_ids,
            "logits": logits,
            "confidence": confidence,
            "steps_log": steps_log,
        }

    def _sample_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample tokens with temperature and optional nucleus (top-p) sampling."""
        cfg = self.config
        B, L, V = logits.shape

        logits = logits / max(cfg.temperature, 1e-8)

        if cfg.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            remove = cum_probs > cfg.top_p
            # Keep at least one token
            remove[..., :1] = False
            # Shift: include the first token that exceeds the threshold
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            indices_to_remove = remove.scatter(-1, sorted_indices, remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        flat = probs.reshape(-1, V)
        sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return sampled.reshape(B, L)

    @staticmethod
    def _unmask_topk(
        target_ids: torch.Tensor,
        predicted: torch.Tensor,
        mask: torch.Tensor,
        score: torch.Tensor,
        n_to_unmask: torch.Tensor,
        high_is_good: bool = True,
    ):
        """Unmask top-k scored tokens (modifies target_ids and mask in-place).

        For confidence: high_is_good=True (unmask most confident).
        For entropy:    high_is_good=False (unmask least entropic).
        """
        B, L = mask.shape

        for b in range(B):
            k = int(n_to_unmask[b].item())
            if k <= 0:
                continue

            if high_is_good:
                _, idx = score[b].topk(k, largest=True)
            else:
                _, idx = score[b].topk(k, largest=False)

            mask[b, idx] = False
            target_ids[b, idx] = predicted[b, idx]

    @staticmethod
    def _unmask_random(
        target_ids: torch.Tensor,
        predicted: torch.Tensor,
        mask: torch.Tensor,
        n_to_unmask: torch.Tensor,
    ):
        """Unmask random subset of masked tokens (modifies in-place)."""
        B, L = mask.shape

        for b in range(B):
            k = int(n_to_unmask[b].item())
            if k <= 0:
                continue

            masked_idx = mask[b].nonzero(as_tuple=True)[0]
            perm = torch.randperm(masked_idx.shape[0], device=mask.device)[:k]
            selected = masked_idx[perm]
            mask[b, selected] = False
            target_ids[b, selected] = predicted[b, selected]


def build_sampler(config: dict) -> MaskedDiffusionSampler:
    """Factory: build sampler from config dict."""
    cfg = SamplerConfig(**config)
    return MaskedDiffusionSampler(cfg)
