#!/usr/bin/env python3
"""
Main training entry point for DFlow-LM.

Usage:
    # Single GPU
    python scripts/train.py --config configs/dflow/dflow_gpt2medium_wikitext.yaml

    # Multi-GPU with Accelerate
    accelerate launch scripts/train.py --config configs/dflow/dflow_gpt2medium_wikitext.yaml

    # 8-GPU launch
    accelerate launch --multi_gpu --num_processes 8 scripts/train.py \
        --config configs/dflow/dflow_gpt2medium_wikitext.yaml

    # Override config keys via CLI
    accelerate launch scripts/train.py --config configs/dflow/dflow_gpt2medium_wikitext.yaml \
        training.max_steps=200000 training.batch_size=16

Config files are YAML; see configs/dflow/ for available configurations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import yaml


def load_config(config_path: str, overrides: list[str] = None) -> dict:
    """Load YAML config with optional CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for kv in overrides:
            key, _, val = kv.partition("=")
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    if val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
            d[keys[-1]] = val
    return cfg


def main():
    parser = argparse.ArgumentParser(description="DFlow-LM Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides for config")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    from models import MaskedDiffusionConfig
    from trainers import MaskedDiffusionTrainer, MaskedDiffusionTrainingConfig

    model_cfg = MaskedDiffusionConfig(**cfg.get("model", {}))
    train_cfg_dict = cfg.get("training", {})
    train_cfg_dict["model"] = model_cfg

    # Type coercions for CLI overrides
    float_fields = {"lr", "weight_decay", "grad_clip", "prompt_ratio"}
    int_fields = {"warmup_steps", "max_steps", "batch_size", "num_workers",
                  "log_every", "eval_every", "save_every", "seed",
                  "accumulation_steps", "max_length", "eval_max_batches",
                  "elbo_time_steps"}
    for k, v in train_cfg_dict.items():
        if k in float_fields and isinstance(v, str):
            train_cfg_dict[k] = float(v)
        elif k in int_fields and isinstance(v, str):
            train_cfg_dict[k] = int(v)

    train_cfg = MaskedDiffusionTrainingConfig(
        **{k: v for k, v in train_cfg_dict.items()
           if k in MaskedDiffusionTrainingConfig.__dataclass_fields__}
    )

    print(f"[train.py] config={args.config}")
    print(f"[train.py] run_name={train_cfg.run_name}")
    print(f"[train.py] output_dir={train_cfg.output_dir}")

    trainer = MaskedDiffusionTrainer(train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
