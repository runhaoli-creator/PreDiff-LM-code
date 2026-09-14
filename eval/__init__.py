"""DFlow-LM Evaluation package."""

from .discrete_eval import compute_elbo_ppl, evaluate_accuracy_by_mask_rate, generate_and_evaluate
from .perplexity import compute_flow_loss, compute_ar_perplexity, compute_pseudo_perplexity
from .generation import (
    distinct_n, repetition_rate, compute_bleu, compute_rouge,
    compute_mauve, code_exact_match, code_prefix_match, generate_samples,
)
from .latency import benchmark_cflow, benchmark_ar, print_latency_table, LatencyResult

__all__ = [
    "compute_elbo_ppl",
    "evaluate_accuracy_by_mask_rate",
    "generate_and_evaluate",
    "compute_flow_loss",
    "compute_ar_perplexity",
    "compute_pseudo_perplexity",
    "distinct_n",
    "repetition_rate",
    "compute_bleu",
    "compute_rouge",
    "compute_mauve",
    "code_exact_match",
    "code_prefix_match",
    "generate_samples",
    "benchmark_cflow",
    "benchmark_ar",
    "print_latency_table",
    "LatencyResult",
]
