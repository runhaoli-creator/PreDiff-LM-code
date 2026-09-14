"""
MaskedDiffusionLM: Discrete Masked Diffusion Language Model with Pretrained Backbone.

Key innovations over standard MDLM (Sahoo et al., 2024):
  1. Pretrained GPT-2 backbone warm-start (vs training from scratch)
  2. Hybrid 4D attention: causal prompt + bidirectional target
  3. Self-conditioning in discrete token space
  4. Mask-rate conditioning via sinusoidal embedding
  5. Designed for conditional (prompt→target) generation

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │  MaskedDiffusionLM                                               │
    │                                                                  │
    │  prompt x  ──embed──► [x₁,…,xₙ]  ──┐                           │
    │                       (causal attn) ├──► GPT2 backbone ──► h    │
    │  masked target  ──embed──────────► [m₁, y₂, m₃,…]              │
    │                    (bidir attn)                                   │
    │                                                                  │
    │  h[target positions] ──► lm_head (tied wte) ──► logits ──► CE   │
    │                                                                  │
    │  Conditioning: mask_rate t ──► SinusoidalEmbed ──► add to target │
    │  Self-cond:    prev x̂₀ ──► self_cond_proj ──► add to target     │
    └──────────────────────────────────────────────────────────────────┘

Training: randomly mask t~schedule fraction of target tokens → predict originals.
Sampling: start all-masked → iteratively unmask (confidence-adaptive or random).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, AutoConfig

from .time_embedding import SinusoidalTimeEmbedding


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class MaskedDiffusionConfig:
    """Configuration for MaskedDiffusionLM."""

    # Backbone
    backbone_name: str = "gpt2-medium"
    freeze_backbone: bool = False
    freeze_backbone_layers: int = 0
    gradient_checkpointing: bool = True

    # Masking schedule for training (distribution over mask rates)
    mask_schedule: str = "cosine"
    """Training mask rate distribution:
       'uniform'     – t ~ U(0,1)
       'cosine'      – t = cos(u * π/2), u ~ U(0,1)  (more weight near t=0)
       'logit_normal' – t = σ(N(0, 1.5))  (bimodal: near 0 and 1)
       'square'       – t = 1 - √u, u ~ U(0,1)  (more weight on low mask rates)
    """

    # Self-conditioning
    self_conditioning: bool = True
    self_cond_prob: float = 0.5

    # Loss configuration
    loss_weighting: str = "uniform"
    """Loss weighting:
       'uniform' – simple mean CE over masked positions
       'elbo'    – weight each sample by 1/(1-t) for proper ELBO
    """
    label_smoothing: float = 0.0

    # Architecture details
    inject_mask_rate: bool = True
    """Condition on mask rate via sinusoidal embedding added to target positions."""
    time_embed_max_period: float = 10000.0

    # Dimensions (auto-detected from backbone if None)
    vocab_size: Optional[int] = None
    max_prompt_len: int = 512
    max_target_len: int = 512


# ──────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────

class MaskedDiffusionLM(nn.Module):
    """Discrete Masked Diffusion Language Model with Pretrained Backbone.

    Operates in discrete token space: randomly masks target tokens during
    training and learns to predict them. No continuous embeddings, no ODE.
    """

    def __init__(self, config: MaskedDiffusionConfig):
        super().__init__()
        self.config = config

        # ── Load backbone ─────────────────────────────────────────────
        backbone_cfg = AutoConfig.from_pretrained(config.backbone_name)
        self.backbone = GPT2Model.from_pretrained(config.backbone_name)

        self.hidden_dim: int = backbone_cfg.n_embd
        self.vocab_size: int = config.vocab_size or backbone_cfg.vocab_size
        self.num_layers: int = backbone_cfg.n_layer

        # ── Token embedding (shared with backbone) ────────────────────
        self.token_embed = self.backbone.wte  # (V, E)

        # ── Learned mask embedding ────────────────────────────────────
        # A single learnable vector that replaces token embeddings at
        # masked positions. Initialized near zero for stable warm-start.
        self.mask_embedding = nn.Parameter(
            torch.randn(self.hidden_dim) * 0.02
        )

        # ── Mask-rate conditioning ────────────────────────────────────
        if config.inject_mask_rate:
            self.mask_rate_embed = SinusoidalTimeEmbedding(
                hidden_dim=self.hidden_dim,
                max_period=config.time_embed_max_period,
                learnable_proj=True,
            )
        else:
            self.mask_rate_embed = None

        # ── LM head (tied to token embeddings) ────────────────────────
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=True)
        self.lm_head.weight = self.token_embed.weight  # tie weights
        # Initialize bias to zero
        nn.init.zeros_(self.lm_head.bias)

        # ── Self-conditioning projection ──────────────────────────────
        if config.self_conditioning:
            self.self_cond_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.init.zeros_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)
        else:
            self.self_cond_proj = None

        # ── Freeze backbone layers if requested ───────────────────────
        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        elif config.freeze_backbone_layers > 0:
            for i, block in enumerate(self.backbone.h):
                if i < config.freeze_backbone_layers:
                    for p in block.parameters():
                        p.requires_grad_(False)

        # ── Gradient checkpointing ────────────────────────────────────
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    # ──────────────────────────────────────────────────────────────────
    # Attention mask
    # ──────────────────────────────────────────────────────────────────

    def _build_attention_mask(
        self,
        L_x: int,
        L_y: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build hybrid 4D attention mask.

        Layout: [prompt₁…prompt_Lx | target₁…target_Ly]
          - Prompt ↔ Prompt: causal (preserve pretrained AR representations)
          - Target → Prompt: full (targets see all prompt context)
          - Target ↔ Target: bidirectional (denoising needs global context)
          - Prompt → Target: blocked (prompt encoding is independent)

        Returns:
            (1, 1, L, L) float mask with 0.0 for attend, -inf for block.
        """
        L = L_x + L_y
        mask = torch.full((1, 1, L, L), float('-inf'), device=device, dtype=dtype)

        # Prompt ↔ Prompt: causal (lower-triangular)
        causal = torch.triu(
            torch.full((L_x, L_x), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        )
        mask[0, 0, :L_x, :L_x] = causal

        # Target → Prompt: full attention
        mask[0, 0, L_x:, :L_x] = 0.0

        # Target ↔ Target: bidirectional
        mask[0, 0, L_x:, L_x:] = 0.0

        # Prompt → Target: remains -inf (blocked)
        return mask

    # ──────────────────────────────────────────────────────────────────
    # Input construction
    # ──────────────────────────────────────────────────────────────────

    def _build_input_embeds(
        self,
        prompt_ids: torch.Tensor,       # (B, Lx)
        target_ids: torch.Tensor,       # (B, Ly)
        mask: torch.Tensor,             # (B, Ly) bool — True = masked
        mask_rate: torch.Tensor,        # (B,) float
        x_self_cond: Optional[torch.Tensor] = None,  # (B, Ly, E)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct input embeddings with masked target tokens.

        Returns:
            inputs_embeds: (B, L, E)
            attention_mask: (1, 1, L, L)
            position_ids:   (1, L)
        """
        B, L_x = prompt_ids.shape
        L_y = target_ids.shape[1]

        # Prompt embeddings
        prompt_emb = self.token_embed(prompt_ids)  # (B, Lx, E)

        # Target embeddings: real tokens at unmasked, mask_embedding at masked
        target_emb_clean = self.token_embed(target_ids)  # (B, Ly, E)
        mask_exp = mask.unsqueeze(-1).float()  # (B, Ly, 1)
        target_emb = (
            target_emb_clean * (1.0 - mask_exp)
            + self.mask_embedding.unsqueeze(0).unsqueeze(0) * mask_exp
        )

        # Self-conditioning: add projected soft predictions at all target positions
        # Always pass through self_cond_proj (even with zeros) to keep it
        # in the computation graph for DDP (avoids unused-parameter errors).
        if self.self_cond_proj is not None:
            sc_input = (
                x_self_cond if x_self_cond is not None
                else torch.zeros_like(target_emb)
            )
            target_emb = target_emb + self.self_cond_proj(sc_input)

        # Mask-rate conditioning: add sinusoidal embedding to ALL target positions
        # This tells the model the overall noise level for calibrated predictions.
        if self.mask_rate_embed is not None:
            rate_emb = self.mask_rate_embed(mask_rate)  # (B, E)
            target_emb = target_emb + rate_emb.unsqueeze(1)

        # Concatenate
        inputs_embeds = torch.cat([prompt_emb, target_emb], dim=1)  # (B, L, E)

        # Custom 4D attention mask
        attention_mask = self._build_attention_mask(
            L_x, L_y, inputs_embeds.device, inputs_embeds.dtype
        )

        # Position IDs — clamped to backbone max (1024 for GPT-2)
        L = L_x + L_y
        max_pos = self.backbone.wpe.num_embeddings
        position_ids = torch.arange(L, device=inputs_embeds.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=max_pos - 1)

        return inputs_embeds, attention_mask, position_ids

    # ──────────────────────────────────────────────────────────────────
    # Mask sampling
    # ──────────────────────────────────────────────────────────────────

    def _sample_mask_rate(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample mask rates from the configured training distribution."""
        sched = self.config.mask_schedule
        if sched == "uniform":
            return torch.rand(B, device=device).clamp(min=0.01, max=0.99)
        elif sched == "cosine":
            u = torch.rand(B, device=device)
            return torch.cos(u * math.pi / 2).clamp(min=0.01, max=0.99)
        elif sched == "logit_normal":
            u = torch.randn(B, device=device) * 1.5
            return torch.sigmoid(u).clamp(min=0.01, max=0.99)
        elif sched == "square":
            u = torch.rand(B, device=device)
            return (1.0 - u.sqrt()).clamp(min=0.01, max=0.99)
        else:
            return torch.rand(B, device=device).clamp(min=0.01, max=0.99)

    def _create_mask(
        self,
        target_ids: torch.Tensor,
        mask_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Create random binary mask for target tokens.

        Args:
            target_ids: (B, Ly) target token IDs.
            mask_rate:  (B,) fraction of tokens to mask.

        Returns:
            (B, Ly) bool mask — True = masked.
        """
        B, L_y = target_ids.shape
        # Per-token probability of masking
        mask = torch.rand(B, L_y, device=target_ids.device) < mask_rate.unsqueeze(1)

        # Ensure at least one token is masked per sample
        no_mask = mask.sum(dim=1) == 0
        if no_mask.any():
            idx = torch.randint(L_y, (no_mask.sum(),), device=mask.device)
            mask[no_mask, idx] = True

        return mask

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        mode: Literal["train", "sample"] = "train",
        mask: Optional[torch.Tensor] = None,
        mask_rate: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            prompt_ids:  (B, Lx) tokenized prompt.
            target_ids:  (B, Ly) tokenized target (original tokens).
                         In sample mode, contains predicted tokens at unmasked positions.
            mode:        "train" | "sample".
            mask:        (B, Ly) bool mask (True = masked). Sampled if None.
            mask_rate:   (B,) float mask rate. Sampled if None.
            x_self_cond: (B, Ly, E) soft self-conditioning embeddings from prior pass.

        Returns dict:
            "loss"         – (train) weighted CE loss
            "ce_loss"      – (train) raw CE at masked positions
            "logits"       – (B, Ly, V) vocab logits at target positions
            "predicted_ids" – (B, Ly) argmax token predictions
            "mask"         – (B, Ly) mask used
            "mask_rate"    – (B,) mask rates used
            "masked_acc"   – (train) accuracy at masked positions
            "unmasked_acc" – (train) accuracy at unmasked positions
            "x_self_cond"  – (B, Ly, E) for self-conditioning in next pass
        """
        B = prompt_ids.shape[0]
        device = prompt_ids.device
        L_y = target_ids.shape[1]
        out = {}

        # ── Sample mask rate and create mask if not provided ──────────
        if mask_rate is None:
            mask_rate = self._sample_mask_rate(B, device)
        if mask is None:
            mask = self._create_mask(target_ids, mask_rate)

        out["mask"] = mask
        out["mask_rate"] = mask_rate

        # ── Build inputs and run backbone ─────────────────────────────
        inputs_embeds, attn_mask, pos_ids = self._build_input_embeds(
            prompt_ids, target_ids, mask, mask_rate, x_self_cond,
        )

        backbone_out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=False,
        )

        hidden = backbone_out.last_hidden_state  # (B, Lx+Ly, E)
        L_x = prompt_ids.shape[1]
        target_hidden = hidden[:, L_x:, :]  # (B, Ly, E)

        # ── Predict logits at all target positions ────────────────────
        logits = self.lm_head(target_hidden)  # (B, Ly, V)
        out["logits"] = logits
        out["predicted_ids"] = logits.argmax(dim=-1)

        # ── Self-conditioning output ──────────────────────────────────
        with torch.no_grad():
            soft_probs = F.softmax(logits.float(), dim=-1)
            out["x_self_cond"] = (
                soft_probs @ self.token_embed.weight.detach().float()
            ).to(logits.dtype)

        # ── Loss computation (train mode) ─────────────────────────────
        if mode == "train":
            V = logits.shape[-1]
            flat_logits = logits.reshape(-1, V)
            flat_targets = target_ids.reshape(-1)
            flat_mask = mask.reshape(-1).float()

            per_token_ce = F.cross_entropy(
                flat_logits, flat_targets,
                reduction='none',
                label_smoothing=self.config.label_smoothing,
            )

            if self.config.loss_weighting == "elbo":
                # ELBO-proper: weight each sample by 1/(1-t)
                per_token_ce_2d = per_token_ce.view(B, L_y)
                mask_f = mask.float()
                per_sample_ce = (
                    (per_token_ce_2d * mask_f).sum(dim=1)
                    / mask_f.sum(dim=1).clamp(min=1)
                )
                weight = 1.0 / (1.0 - mask_rate + 1e-5)
                ce_loss = (per_sample_ce * weight).mean()
            else:
                # Uniform: simple mean over masked positions
                ce_loss = (
                    (per_token_ce * flat_mask).sum()
                    / flat_mask.sum().clamp(min=1)
                )

            out["loss"] = ce_loss
            out["ce_loss"] = ce_loss

            # ── Accuracy metrics ──────────────────────────────────────
            with torch.no_grad():
                correct = (logits.argmax(-1) == target_ids).float()
                mask_f = mask.float()
                out["masked_acc"] = (
                    (correct * mask_f).sum() / mask_f.sum().clamp(min=1)
                )
                unmask_f = (~mask).float()
                out["unmasked_acc"] = (
                    (correct * unmask_f).sum() / unmask_f.sum().clamp(min=1)
                )

        return out

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"MaskedDiffusionLM(backbone={self.config.backbone_name}, "
            f"trainable_params={n:,}, "
            f"schedule={self.config.mask_schedule}, "
            f"self_cond={self.config.self_conditioning})"
        )
