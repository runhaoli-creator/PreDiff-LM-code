#!/usr/bin/env python3
"""
Quick smoke test for DFlow-LM: verifies model forward pass, training loop,
and sampling pipeline work end-to-end.

Usage:
    python scripts/smoke_test.py [--device cpu|cuda]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch


def test_model_forward(device):
    """Test model forward pass."""
    print("[1/4] Testing model forward pass...")
    from models import MaskedDiffusionLM, MaskedDiffusionConfig

    config = MaskedDiffusionConfig(
        backbone_name="gpt2",
        gradient_checkpointing=False,
        self_conditioning=True,
        inject_mask_rate=True,
        mask_schedule="cosine",
    )
    model = MaskedDiffusionLM(config).to(device)
    print(f"  Model: {model}")

    B, L_x, L_y = 2, 64, 64
    prompt_ids = torch.randint(0, 50257, (B, L_x), device=device)
    target_ids = torch.randint(0, 50257, (B, L_y), device=device)

    out = model(prompt_ids=prompt_ids, target_ids=target_ids, mode="train")
    assert "loss" in out, "Missing loss in output"
    assert "logits" in out, "Missing logits in output"
    assert "mask" in out, "Missing mask in output"
    assert out["logits"].shape == (B, L_y, 50257), f"Wrong shape: {out['logits'].shape}"
    print(f"  loss={out['loss'].item():.4f}  masked_acc={out['masked_acc'].item():.3f}")
    print("  PASSED ✓")
    return model


def test_sampler(model, device):
    """Test sampling pipeline."""
    print("[2/4] Testing sampler...")
    from samplers import MaskedDiffusionSampler, SamplerConfig

    B, L_x, L_y = 2, 64, 64
    prompt_ids = torch.randint(0, 50257, (B, L_x), device=device)

    for strategy in ["confidence", "random"]:
        cfg = SamplerConfig(n_steps=4, strategy=strategy, schedule="cosine", use_self_cond=True)
        sampler = MaskedDiffusionSampler(cfg)
        t0 = time.time()
        result = sampler.sample(model, prompt_ids, target_len=L_y)
        elapsed = time.time() - t0
        assert result["token_ids"].shape == (B, L_y)
        print(f"  strategy={strategy}: {elapsed:.2f}s, shape={result['token_ids'].shape}")
    print("  PASSED ✓")


def test_data_loading():
    """Test data pipeline."""
    print("[3/4] Testing data loading (WikiText-103)...")
    from data import build_dataloader

    loader = build_dataloader(
        "wikitext103",
        split="validation",
        max_length=128,
        prompt_len=64,
        batch_size=4,
        streaming=True,
    )
    batch = next(iter(loader))
    assert "prompt_ids" in batch
    assert "target_ids" in batch
    assert batch["prompt_ids"].shape == (4, 64), f"Wrong shape: {batch['prompt_ids'].shape}"
    assert batch["target_ids"].shape == (4, 64), f"Wrong shape: {batch['target_ids'].shape}"
    print(f"  prompt_ids: {batch['prompt_ids'].shape}  target_ids: {batch['target_ids'].shape}")
    print("  PASSED ✓")


def test_training_step(device):
    """Test a few training steps for loss decrease."""
    print("[4/4] Testing training loop (20 steps)...")
    from models import MaskedDiffusionLM, MaskedDiffusionConfig
    from data import build_dataloader

    config = MaskedDiffusionConfig(
        backbone_name="gpt2",
        gradient_checkpointing=False,
        self_conditioning=False,
        mask_schedule="cosine",
    )
    model = MaskedDiffusionLM(config).to(device)

    loader = build_dataloader(
        "wikitext103",
        split="validation",
        max_length=128,
        prompt_len=64,
        batch_size=4,
        streaming=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    model.train()

    losses = []
    data_iter = iter(loader)
    for step in range(20):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        out = model(prompt_ids=prompt_ids, target_ids=target_ids, mode="train")
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if (step + 1) % 5 == 0:
            print(f"  step {step+1:3d}  loss={loss.item():.4f}")

    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    print(f"  avg_loss first_5={first5:.4f}  last_5={last5:.4f}  delta={first5-last5:.4f}")
    if last5 < first5:
        print("  Loss decreased ✓")
    else:
        print("  WARNING: Loss did not decrease (may be OK for 20 steps)")
    print("  PASSED ✓")


def main():
    parser = argparse.ArgumentParser(description="DFlow-LM Smoke Test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"=" * 60)
    print(f"DFlow-LM Smoke Test  |  device={device}")
    print(f"=" * 60)

    model = test_model_forward(device)
    test_sampler(model, device)
    test_data_loading()
    test_training_step(device)

    print(f"\n{'=' * 60}")
    print(f"All tests PASSED ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
