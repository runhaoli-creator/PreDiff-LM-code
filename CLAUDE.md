# DFlow-LM

## Overview
- **What**: Discrete Flow Matching for Language Modeling — research project
- **Repo**: local

## Tech Stack
- Python, PyTorch, Transformers, WandB

## Project Structure
```
├── models/        # source
├── trainers/      # training logic
├── samplers/      # sampling methods
├── eval/          # evaluation
├── data/          # datasets
├── configs/       # experiment configs
├── scripts/       # training & eval scripts
├── paper/         # paper source (do not touch)
├── paper_runhao/  # paper edits (edit here)
```

## Key Commands
```bash
# setup environment
bash setup_env.sh

# smoke test
python scripts/smoke_test.py

# train
python scripts/train.py

# evaluate
python scripts/eval.py
```

## Paper Rules
- **Only edit**: `paper_runhao/main.tex` and files in `paper_runhao/`
- **Never touch**: anything in `paper/` directory

## Code Conventions
- Follow existing style in this repo
- Config changes go in configs/, not hardcoded
- Every experiment must log: config, seed, git hash
- Results files go in results/ (git tracked), large files in outputs/ (gitignored)

## Current Focus
Writing phase — drafting and revising paper in paper_runhao/
