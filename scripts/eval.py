#!/usr/bin/env python3
"""
Evaluation script for DFlow-LM.

Runs:
  1. ELBO-based NLL/PPL estimation
  2. Token accuracy at various mask rates
  3. Generation quality: distinct-n, BLEU, ROUGE
  4. Latency benchmarks

Usage:
    python scripts/eval.py \
        --ckpt runs/dflow/dflow_gpt2medium_wikitext/checkpoints/best \
        --dataset wikitext103 \
        --n_steps 16 32 64

    # Quick eval (fewer batches)
    python scripts/eval.py \
        --ckpt runs/dflow/dflow_gpt2medium_wikitext/checkpoints/best \
        --max_eval_batches 20 --n_gen_samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="DFlow-LM Evaluation")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint dir or .pt file")
    p.add_argument("--dataset", type=str, default="wikitext103",
                   choices=["wikitext103", "pg19", "codeparrot"])
    p.add_argument("--n_steps", type=int, nargs="+", default=[16, 32, 64],
                   help="Number of denoising steps")
    p.add_argument("--strategy", type=str, default="confidence",
                   choices=["confidence", "random", "entropy"],
                   help="Unmasking strategy")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_eval_batches", type=int, default=100)
    p.add_argument("--n_gen_samples", type=int, default=200)
    p.add_argument("--elbo_time_steps", type=int, default=64)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer", type=str, default="gpt2")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.ckpt).parent.parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"dataset": args.dataset, "metrics": {}}

    # ── Load model ──────────────────────────────────────────────────
    from models import MaskedDiffusionLM, MaskedDiffusionConfig

    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_config = ckpt.get("config", {})
    model_cfg = MaskedDiffusionConfig(**ckpt_config.get("model", {}))
    model = MaskedDiffusionLM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded DFlow-LM from {ckpt_path} (step {ckpt.get('global_step', '?')})")

    # ── Load data ───────────────────────────────────────────────────
    from data import build_dataloader

    val_split = "validation" if args.dataset != "codeparrot" else "valid"
    val_loader = build_dataloader(
        args.dataset,
        split=val_split,
        max_length=512,
        prompt_len=256,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # ── ELBO PPL ────────────────────────────────────────────────────
    from eval.discrete_eval import compute_elbo_ppl, evaluate_accuracy_by_mask_rate

    print("\n=== ELBO PPL ===")
    elbo = compute_elbo_ppl(
        model, val_loader,
        n_time_steps=args.elbo_time_steps,
        max_batches=args.max_eval_batches,
        device=device,
    )
    print(f"  NLL: {elbo['nll']:.4f}  PPL: {elbo['ppl']:.2f}")
    results["metrics"]["elbo"] = elbo

    # ── Accuracy by mask rate ───────────────────────────────────────
    print("\n=== Accuracy by Mask Rate ===")
    acc_results = evaluate_accuracy_by_mask_rate(
        model, val_loader,
        mask_rates=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        max_batches=min(50, args.max_eval_batches),
        device=device,
    )
    for rate, metrics in acc_results.items():
        print(f"  mask_rate={rate:.1f}  acc={metrics['accuracy']:.3f}  ce={metrics['ce_loss']:.4f}")
    results["metrics"]["accuracy_by_mask_rate"] = {str(k): v for k, v in acc_results.items()}

    # ── Generation quality at different step counts ─────────────────
    from samplers import MaskedDiffusionSampler, SamplerConfig
    from eval.generation import distinct_n, repetition_rate, compute_bleu, compute_rouge

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for n_steps in args.n_steps:
        print(f"\n=== Generation ({n_steps} steps, {args.strategy}) ===")
        sampler = MaskedDiffusionSampler(SamplerConfig(
            n_steps=n_steps,
            strategy=args.strategy,
            schedule="cosine",
            use_self_cond=True,
        ))

        references, hypotheses = [], []
        gen_count = 0
        for batch in val_loader:
            if gen_count >= args.n_gen_samples:
                break
            prompt_ids = batch["prompt_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            B, L_y = target_ids.shape

            result = sampler.sample(model, prompt_ids, target_len=L_y)
            gen_ids = result["token_ids"]

            for b in range(min(B, args.n_gen_samples - gen_count)):
                ref = tokenizer.decode(target_ids[b].cpu().tolist(), skip_special_tokens=True)
                hyp = tokenizer.decode(gen_ids[b].cpu().tolist(), skip_special_tokens=True)
                references.append(ref)
                hypotheses.append(hyp)
                gen_count += 1

        hyp_ids = [tokenizer.encode(h) for h in hypotheses]
        gen_metrics = {
            "distinct1": distinct_n(hyp_ids, 1),
            "distinct2": distinct_n(hyp_ids, 2),
            "repetition": repetition_rate(hyp_ids),
            **compute_bleu(references, hypotheses),
            **compute_rouge(references, hypotheses),
        }
        print(f"  D1={gen_metrics['distinct1']:.3f}  D2={gen_metrics['distinct2']:.3f}  "
              f"Rep={gen_metrics['repetition']:.3f}  BLEU={gen_metrics.get('bleu', float('nan')):.2f}")
        results["metrics"][f"generation_{n_steps}steps"] = gen_metrics

    # ── Save results ────────────────────────────────────────────────
    out_path = output_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
