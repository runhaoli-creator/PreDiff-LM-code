"""
Masked Diffusion Trainer: full training loop for MaskedDiffusionLM.

Uses HuggingFace Accelerate for distributed training (DDP), mixed precision,
gradient checkpointing, and automatic device placement.

Supports:
  - Self-conditioning with configurable probability
  - ELBO-based evaluation for proper NLL/PPL estimates
  - Generation evaluation (BLEU, distinct-n) via sampler
  - Comprehensive logging and checkpointing
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from models.masked_diffusion_lm import MaskedDiffusionLM, MaskedDiffusionConfig
from data import build_dataloader


@dataclass
class MaskedDiffusionTrainingConfig:
    """All hyper-parameters for a masked diffusion training run."""

    # Run identity
    run_name: str = "dflow_lm_run"
    output_dir: str = "runs/dflow"
    seed: int = 42

    # Model
    model: MaskedDiffusionConfig = field(default_factory=MaskedDiffusionConfig)

    # Data
    dataset: str = "wikitext103"
    max_length: int = 512
    prompt_ratio: float = 0.5
    batch_size: int = 16
    num_workers: int = 0

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 2000
    max_steps: int = 200_000
    grad_clip: float = 1.0
    accumulation_steps: int = 1

    # Logging & evaluation
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 10000
    use_wandb: bool = False
    eval_max_batches: int = 50
    elbo_time_steps: int = 64

    # Mixed precision
    mixed_precision: str = "bf16"


class MaskedDiffusionTrainer:
    """Training loop for MaskedDiffusionLM.

    Args:
        cfg: MaskedDiffusionTrainingConfig dataclass.
    """

    def __init__(self, cfg: MaskedDiffusionTrainingConfig):
        self.cfg = cfg

        # Accelerator
        self.accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=cfg.accumulation_steps,
            log_with="wandb" if cfg.use_wandb else None,
        )

        set_seed(cfg.seed)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save run provenance
        self._save_provenance()

        # Build model
        self.model = MaskedDiffusionLM(cfg.model)
        self.accelerator.print(f"Model: {self.model}")
        self.accelerator.print(f"Trainable params: {self.model.num_parameters():,}")

        # Build dataloaders
        prompt_len = int(cfg.max_length * cfg.prompt_ratio)
        self.train_loader = build_dataloader(
            cfg.dataset,
            split="train",
            max_length=cfg.max_length,
            prompt_len=prompt_len,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            streaming=True,
        )
        self.val_loader = build_dataloader(
            cfg.dataset,
            split="validation" if cfg.dataset != "codeparrot" else "valid",
            max_length=cfg.max_length,
            prompt_len=prompt_len,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            streaming=True,
        )

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(param_groups, lr=cfg.lr)

        # LR schedule: linear warmup then cosine decay
        warmup = LinearLR(
            self.optimizer, start_factor=1e-6, end_factor=1.0,
            total_iters=cfg.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=max(cfg.max_steps - cfg.warmup_steps, 1),
            eta_min=cfg.lr * 0.05,
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[cfg.warmup_steps],
        )

        # Accelerate wrapping
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ──────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self):
        """Run the full training loop."""
        self.model.train()
        train_iter = iter(self.train_loader)
        pbar = tqdm(
            total=self.cfg.max_steps,
            disable=not self.accelerator.is_main_process,
            desc="Training",
        )

        running_loss = 0.0
        running_masked_acc = 0.0
        t0 = time.time()

        while self.global_step < self.cfg.max_steps:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            prompt_ids = batch["prompt_ids"]
            target_ids = batch["target_ids"]

            with self.accelerator.accumulate(self.model):
                # ── Self-conditioning ─────────────────────────────────
                x_self_cond = None
                cached_mask = None
                cached_mask_rate = None
                model_unwrapped = self.accelerator.unwrap_model(self.model)
                use_self_cond = model_unwrapped.config.self_conditioning

                if use_self_cond and torch.rand(1).item() < model_unwrapped.config.self_cond_prob:
                    with torch.no_grad():
                        first_out = self.model(
                            prompt_ids=prompt_ids,
                            target_ids=target_ids,
                            mode="train",
                            x_self_cond=None,
                        )
                        x_self_cond = first_out["x_self_cond"].detach()
                        cached_mask = first_out["mask"].detach()
                        cached_mask_rate = first_out["mask_rate"].detach()

                # ── Main forward pass ─────────────────────────────────
                out = self.model(
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    mode="train",
                    mask=cached_mask,
                    mask_rate=cached_mask_rate,
                    x_self_cond=x_self_cond,
                )
                loss = out["loss"]

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()
            running_masked_acc += out.get("masked_acc", torch.tensor(0.0)).item()
            self.global_step += 1
            pbar.update(1)

            # ── Logging ───────────────────────────────────────────────
            if (
                self.global_step % self.cfg.log_every == 0
                and self.accelerator.is_main_process
            ):
                avg_loss = running_loss / self.cfg.log_every
                avg_acc = running_masked_acc / self.cfg.log_every
                elapsed = time.time() - t0
                tokens_per_step = self.cfg.batch_size * (self.cfg.max_length // 2)
                tps = self.cfg.log_every * tokens_per_step / max(elapsed, 1e-6)
                lr_now = self.scheduler.get_last_lr()[0]

                self.accelerator.print(
                    f"[{self.global_step}/{self.cfg.max_steps}] "
                    f"loss={avg_loss:.4f}  "
                    f"masked_acc={avg_acc:.3f}  "
                    f"lr={lr_now:.2e}  "
                    f"tok/s={tps:.0f}"
                )

                if self.cfg.use_wandb:
                    self.accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/masked_acc": avg_acc,
                            "train/lr": lr_now,
                        },
                        step=self.global_step,
                    )

                running_loss = 0.0
                running_masked_acc = 0.0
                t0 = time.time()

            # ── Evaluation ────────────────────────────────────────────
            if self.global_step % self.cfg.eval_every == 0:
                val_loss, val_acc = self.evaluate()
                if self.accelerator.is_main_process:
                    ppl = math.exp(min(val_loss, 20.0))
                    self.accelerator.print(
                        f"  ► val_loss={val_loss:.4f}  "
                        f"val_masked_acc={val_acc:.3f}  "
                        f"val_ppl≈{ppl:.1f}"
                    )
                    if self.cfg.use_wandb:
                        self.accelerator.log(
                            {
                                "val/loss": val_loss,
                                "val/masked_acc": val_acc,
                                "val/ppl": ppl,
                            },
                            step=self.global_step,
                        )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best")

            # ── Checkpoint ────────────────────────────────────────────
            if self.global_step % self.cfg.save_every == 0:
                self.save_checkpoint(f"step_{self.global_step:07d}")

        pbar.close()
        self.save_checkpoint("final")
        self.accelerator.print("Training complete.")

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> tuple[float, float]:
        """Compute mean CE loss and accuracy on the validation set.

        Returns:
            (avg_loss, avg_masked_acc)
        """
        max_batches = max_batches or self.cfg.eval_max_batches
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            out = self.model(
                prompt_ids=batch["prompt_ids"],
                target_ids=batch["target_ids"],
                mode="train",
            )
            total_loss += out["loss"].item()
            total_acc += out.get("masked_acc", torch.tensor(0.0)).item()
            n_batches += 1

        self.model.train()
        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)

        # All-reduce across processes
        stats = torch.tensor([avg_loss, avg_acc], device=self.accelerator.device)
        stats = self.accelerator.reduce(stats, reduction="mean")
        return stats[0].item(), stats[1].item()

    @torch.no_grad()
    def compute_elbo(
        self,
        max_batches: int = 50,
        n_time_steps: int = 64,
    ) -> dict:
        """Compute ELBO-based NLL and perplexity estimate.

        Uses numerical integration over mask rates:
            NLL ≈ (1/T) Σ_t [1/(1-t)] * E[CE at mask_rate=t]

        Returns:
            {"nll": float, "ppl": float}
        """
        self.model.eval()
        device = self.accelerator.device
        model = self.accelerator.unwrap_model(self.model)

        total_weighted_ce = 0.0
        total_tokens = 0

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            prompt_ids = batch["prompt_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            B, L_y = target_ids.shape

            batch_nll = 0.0
            for t_idx in range(n_time_steps):
                t_val = (t_idx + 0.5) / n_time_steps  # midpoint rule
                mask_rate = torch.full((B,), t_val, device=device)
                mask = torch.rand(B, L_y, device=device) < t_val

                # Ensure at least one token masked
                no_mask = mask.sum(dim=1) == 0
                if no_mask.any():
                    idx = torch.randint(L_y, (no_mask.sum(),), device=device)
                    mask[no_mask, idx] = True

                out = self.model(
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    mode="train",
                    mask=mask,
                    mask_rate=mask_rate,
                )

                # CE per masked token
                ce = out["ce_loss"].item()
                # ELBO weight: 1/(1-t) * dt
                weight = (1.0 / max(1.0 - t_val, 1e-5)) * (1.0 / n_time_steps)
                batch_nll += ce * weight

            total_weighted_ce += batch_nll * B * L_y
            total_tokens += B * L_y

        self.model.train()
        avg_nll = total_weighted_ce / max(total_tokens, 1)
        ppl = math.exp(min(avg_nll, 50.0))

        return {"nll": avg_nll, "ppl": ppl}

    # ──────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str):
        """Save model + optimizer + config."""
        if not self.accelerator.is_main_process:
            return

        ckpt_dir = self.output_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(
            {
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "config": asdict(self.cfg),
            },
            ckpt_dir / "checkpoint.pt",
        )
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        self.accelerator.print(f"  Saved checkpoint to {ckpt_dir}")

    @classmethod
    def load_checkpoint(
        cls, run_dir: str, tag: str = "best"
    ) -> tuple[MaskedDiffusionLM, dict]:
        """Load a saved checkpoint.

        Returns:
            model: Loaded MaskedDiffusionLM.
            meta:  Dict with step, config, etc.
        """
        ckpt_path = Path(run_dir) / "checkpoints" / tag / "checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg = MaskedDiffusionConfig(**ckpt["config"]["model"])
        model = MaskedDiffusionLM(cfg)
        model.load_state_dict(ckpt["model_state_dict"])

        return model, {"step": ckpt["global_step"], "config": ckpt["config"]}

    # ──────────────────────────────────────────────────────────────────
    # Provenance
    # ──────────────────────────────────────────────────────────────────

    def _save_provenance(self):
        """Write run metadata for reproducibility."""
        if not self.accelerator.is_main_process:
            return

        git_hash = "unknown"
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            pass

        meta = {
            "run_name": self.cfg.run_name,
            "seed": self.cfg.seed,
            "git_commit": git_hash,
            "config": asdict(self.cfg),
        }
        with open(self.output_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
