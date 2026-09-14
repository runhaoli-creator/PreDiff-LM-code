# DFlow-LM: Discrete Flow Matching for Language Modeling

**Pretrained Language Model Backbone Meets Masked Diffusion**

DFlow-LM is a discrete masked diffusion language model that leverages pretrained GPT-2 backbones for efficient and high-quality text generation. Unlike autoregressive (AR) models that generate tokens sequentially, DFlow-LM generates all target tokens in parallel through iterative denoising, achieving competitive quality with significantly faster convergence.

---

## Key Features

- **Pretrained Backbone Warm-Start**: Initializes from GPT-2 (Medium/Large) for 40× faster convergence compared to training from scratch  
- **Hybrid 4D Attention**: Causal attention for the prompt, bidirectional attention for masked targets  
- **Confidence-Adaptive Unmasking (CAU)**: Unmasks highest-confidence tokens first during iterative sampling  
- **Self-Conditioning**: Feeds previous-step predictions back into the model for improved coherence  
- **Cosine Mask Schedule**: Concentrated training on low mask rates for better conditional quality  
- **Flexible Generation**: Supports unconditional generation, conditional generation, and text infilling  

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  MaskedDiffusionLM                                               │
│                                                                  │
│  prompt x  ──embed──► [x₁,…,xₙ]  ──┐                           │
│                       (causal attn) ├──► GPT-2 backbone ──► h    │
│  masked target  ──embed──────────► [m₁, y₂, m₃,…]              │
│                    (bidir attn)                                   │
│                                                                  │
│  h[target positions] ──► lm_head (tied wte) ──► logits ──► CE   │
│                                                                  │
│  Conditioning: mask_rate t ──► SinusoidalEmbed ──► add to target │
│  Self-cond:    prev x̂₀ ──► self_cond_proj ──► add to target     │
└──────────────────────────────────────────────────────────────────┘
```

**Training:** Randomly mask a fraction `t ~ cosine schedule` of target tokens → predict originals via cross-entropy.  
**Sampling:** Start all-masked → iteratively unmask using confidence-adaptive selection over `N` steps.

---

## Repository Structure

```
DFlow-LM/
├── models/
│   ├── masked_diffusion_lm.py   # MaskedDiffusionLM model
│   └── time_embedding.py        # Sinusoidal + MLP time embedding
├── samplers/
│   └── masked_diffusion_sampler.py  # Iterative unmasking (CAU / random / entropy)
├── trainers/
│   └── masked_diffusion_trainer.py  # Training loop with HF Accelerate
├── eval/
│   ├── discrete_eval.py         # ELBO PPL, accuracy, generation eval
│   ├── perplexity.py            # Flow loss & AR perplexity
│   ├── generation.py            # BLEU, ROUGE, MAUVE, distinct-n
│   └── latency.py               # Throughput benchmarks
├── data/
│   ├── wikitext.py              # WikiText-103 pipeline
│   ├── pg19.py                  # PG-19 pipeline
│   └── codeparrot.py            # CodeParrot pipeline
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── eval.py                  # Evaluation entry point
│   └── smoke_test.py            # Quick verification
├── configs/
│   └── dflow/
│       ├── dflow_gpt2medium_wikitext.yaml   # GPT-2 Medium config
│       ├── dflow_gpt2large_wikitext.yaml    # GPT-2 Large config
│       └── dflow_smoke_test.yaml            # Quick smoke test config
├── paper/
│   ├── main.tex                 # Full paper (NeurIPS format)
│   ├── tables.tex               # All experiment tables
│   ├── generate_figures.py      # Figure generation script
│   └── figures/                 # Publication figures (PDF)
├── requirements.txt
├── setup_env.sh
├── LICENSE
└── README.md
```

---

## Setup

### Option 1: Quick Setup
```bash
# Create conda environment
conda create -n dflow_lm python=3.10 -y
conda activate dflow_lm

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Option 2: One-Shot Script
```bash
bash setup_env.sh
```

### Verify Installation
```bash
conda activate dflow_lm
python scripts/smoke_test.py
```

---

## Training

### Single GPU
```bash
python scripts/train.py --config configs/dflow/dflow_gpt2medium_wikitext.yaml
```

### Multi-GPU (Accelerate)
```bash
# Configure accelerate (first time only)
accelerate config

# Launch on 8 GPUs
accelerate launch --multi_gpu --num_processes 8 \
    scripts/train.py --config configs/dflow/dflow_gpt2medium_wikitext.yaml
```

### Override Config via CLI
```bash
accelerate launch scripts/train.py \
    --config configs/dflow/dflow_gpt2medium_wikitext.yaml \
    training.max_steps=200000 training.batch_size=32 training.lr=5e-5
```

### Quick Smoke Test (200 steps)
```bash
python scripts/train.py --config configs/dflow/dflow_smoke_test.yaml
```

---

## Evaluation

```bash
# Full evaluation (ELBO PPL + generation quality + accuracy)
python scripts/eval.py \
    --ckpt runs/dflow/dflow_gpt2medium_wikitext/checkpoints/best \
    --dataset wikitext103 \
    --n_steps 16 32 64

# Quick evaluation
python scripts/eval.py \
    --ckpt runs/dflow/dflow_gpt2medium_wikitext/checkpoints/best \
    --max_eval_batches 20 --n_gen_samples 50
```

---

## Model Configurations

| Config | Backbone | Params | Effective Batch | Steps |
|--------|----------|--------|----------------|-------|
| `dflow_gpt2medium_wikitext` | GPT-2 Medium | 364M | 256 | 200K |
| `dflow_gpt2large_wikitext` | GPT-2 Large | 783M | 256 | 200K |
| `dflow_smoke_test` | GPT-2 Medium | 364M | 8 | 200 |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_schedule` | `cosine` | Training mask rate distribution |
| `self_conditioning` | `true` | Feed previous predictions back |
| `self_cond_prob` | `0.5` | Probability of using self-conditioning |
| `inject_mask_rate` | `true` | Condition model on current mask rate |
| `loss_weighting` | `uniform` | Loss weighting scheme |
| `lr` | `5e-5` | Learning rate |
| `warmup_steps` | `2000` | Linear warmup steps |
| `grad_clip` | `1.0` | Gradient clipping norm |

---

## Sampling Strategies

DFlow-LM supports three unmasking strategies during generation:

| Strategy | Description |
|----------|-------------|
| `confidence` | **CAU**: Unmask highest-confidence tokens first (default, best quality) |
| `random` | Unmask random tokens (baseline) |
| `entropy` | Unmask lowest-entropy tokens first |

And three mask-rate schedules:

| Schedule | Description |
|----------|-------------|
| `cosine` | Slower unmasking near completion (default) |
| `linear` | Uniform unmasking rate |
| `sqrt` | Faster start, slower finish |

---

## Citation

```bibtex
@article{dflowlm2025,
  title   = {DFlow-LM: Discrete Masked Diffusion Language Modeling with Pretrained Backbone},
  author  = {Zhengtao Yao},
  year    = {2025},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
