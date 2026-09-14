"""
Latency and throughput benchmarking for CFlow-LM vs baselines.

Reports wall-clock time, tokens/sec, and end-to-end latency to produce
512 and 1024 tokens under different ODE step counts {4, 8, 16}.

This is the core empirical claim: few-step flow generation can be
dramatically faster than many-step diffusion at matched quality.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LatencyResult:
    """Results from a single latency benchmark run."""
    model_name: str
    target_len: int
    n_steps: int
    batch_size: int
    wall_time_s: float
    tokens_per_sec: float
    steps_timing: list[float] = field(default_factory=list)
    memory_mb: float = 0.0

    def __repr__(self) -> str:
        return (
            f"LatencyResult(model={self.model_name}, target_len={self.target_len}, "
            f"n_steps={self.n_steps}, tok/s={self.tokens_per_sec:.1f}, "
            f"wall={self.wall_time_s*1000:.1f}ms)"
        )


def _get_memory_mb(device: torch.device) -> float:
    """Return current GPU memory allocation in MB."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return 0.0


@torch.no_grad()
def benchmark_cflow(
    model: nn.Module,
    sampler,
    prompt_ids: torch.Tensor,
    target_len: int = 512,
    n_warmup: int = 2,
    n_runs: int = 10,
) -> LatencyResult:
    """Benchmark CFlow-LM ODE sampling latency.

    Args:
        model:      FlowDecoderLM instance (on target device).
        sampler:    ODESampler with n_steps pre-configured.
        prompt_ids: (B, L_x) prompt tensor.
        target_len: Number of target tokens to generate.
        n_warmup:   Warm-up runs (excluded from timing).
        n_runs:     Timed runs.

    Returns:
        LatencyResult with mean timings.
    """
    device = next(model.parameters()).device
    model.eval()
    B = prompt_ids.shape[0]

    # Warmup
    for _ in range(n_warmup):
        _ = sampler.sample(model, prompt_ids, target_len)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    wall_times = []
    steps_timing_all = []

    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = sampler.sample_with_timing(model, prompt_ids, target_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_times.append(result["wall_time_s"])
        steps_timing_all.append(result["steps_timing"])

    mean_wall = sum(wall_times) / len(wall_times)
    tokens_per_sec = (B * target_len) / mean_wall

    # Mean per-step timing
    n_steps = len(steps_timing_all[0])
    mean_steps = [
        sum(r[i] for r in steps_timing_all) / n_runs for i in range(n_steps)
    ]

    mem_mb = _get_memory_mb(device)

    model_name = getattr(model, "config", None)
    model_name_str = (
        model_name.backbone_name if hasattr(model_name, "backbone_name") else "cflow"
    )

    return LatencyResult(
        model_name=f"CFlow({model_name_str}, {sampler.n_steps}step)",
        target_len=target_len,
        n_steps=sampler.n_steps,
        batch_size=B,
        wall_time_s=mean_wall,
        tokens_per_sec=tokens_per_sec,
        steps_timing=mean_steps,
        memory_mb=mem_mb,
    )


@torch.no_grad()
def benchmark_ar(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    target_len: int = 512,
    n_warmup: int = 2,
    n_runs: int = 5,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> LatencyResult:
    """Benchmark AR language model generation latency (HF CausalLM interface).

    Args:
        model:      HF CausalLM (GPT-2, etc.).
        prompt_ids: (B, L_x) prompt tensor.
        target_len: Number of new tokens to generate.
        n_warmup:   Warm-up runs.
        n_runs:     Timed runs.
        do_sample:  Use sampling (vs greedy).

    Returns:
        LatencyResult.
    """
    device = next(model.parameters()).device
    model.eval()
    B = prompt_ids.shape[0]

    gen_kwargs = {
        "max_new_tokens": target_len,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": model.config.eos_token_id,
    }

    # Warmup
    for _ in range(n_warmup):
        _ = model.generate(prompt_ids, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    wall_times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(prompt_ids, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)

    mean_wall = sum(wall_times) / len(wall_times)
    tokens_per_sec = (B * target_len) / mean_wall

    return LatencyResult(
        model_name=f"AR({getattr(model.config, '_name_or_path', 'gpt2')})",
        target_len=target_len,
        n_steps=target_len,  # AR generates one token at a time
        batch_size=B,
        wall_time_s=mean_wall,
        tokens_per_sec=tokens_per_sec,
        memory_mb=_get_memory_mb(device),
    )


def print_latency_table(results: list[LatencyResult]) -> None:
    """Pretty-print a comparison table of latency results."""
    header = f"{'Model':<50} {'Target':<8} {'Steps':<7} {'Wall(ms)':<10} {'Tok/s':<10} {'Mem(MB)':<10}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.tokens_per_sec, reverse=True):
        print(
            f"{r.model_name:<50} "
            f"{r.target_len:<8} "
            f"{r.n_steps:<7} "
            f"{r.wall_time_s*1000:<10.1f} "
            f"{r.tokens_per_sec:<10.1f} "
            f"{r.memory_mb:<10.1f}"
        )
