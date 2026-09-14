#!/usr/bin/env bash
# setup_env.sh – One-shot environment setup for DFlow-LM
# Usage: bash setup_env.sh

set -e

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> Creating conda environment: dflow_lm (Python 3.10)"
conda create -n dflow_lm python=3.10 -y

echo "==> Activating environment"
conda activate dflow_lm

echo "==> Installing PyTorch (CUDA 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing project dependencies"
pip install -r requirements.txt

echo "==> Verifying installation"
python -c "
import torch
import transformers
import accelerate
import datasets
print(f'torch={torch.__version__}')
print(f'transformers={transformers.__version__}')
print(f'accelerate={accelerate.__version__}')
print(f'datasets={datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "======================================================"
echo "  Environment ready!  conda activate dflow_lm"
echo "======================================================"
