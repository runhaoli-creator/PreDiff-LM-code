# PreDiff-LM

**Pretrained Discrete Masked Diffusion Language Modeling with Hybrid Attention**

[Paper](https://arxiv.org/abs/2607.25157) · [OpenReview](https://openreview.net/forum?id=zl9y14yJuN) · [Model](models/masked_diffusion_lm.py) · [Training](trainers/masked_diffusion_trainer.py)

Public source snapshot shared by Runhao Li. The implementation adapts a pretrained GPT-2 backbone for masked diffusion, with causal attention for prompts, bidirectional attention for targets, and iterative confidence-based unmasking.

## Start here

- **Model:** [`models/masked_diffusion_lm.py`](models/masked_diffusion_lm.py)
- **Sampler:** [`samplers/masked_diffusion_sampler.py`](samplers/masked_diffusion_sampler.py)
- **Training:** [`scripts/train.py`](scripts/train.py) and [`trainers/masked_diffusion_trainer.py`](trainers/masked_diffusion_trainer.py)
- **Configurations:** [`configs/dflow/`](configs/dflow/)
- **Setup and usage:** [Original development README](README_DFlow.md)

## Provenance and version

This code snapshot originates from [zhengtaoyao/DFlow-LM](https://github.com/zhengtaoyao/DFlow-LM), the project's development repository. Original source files, paper drafts, and the MIT license are retained. The upstream repository may require access.

The snapshot retains the earlier DFlow-LM naming and development configurations. Consult the linked PreDiff-LM paper for the reported experimental setup; the imported snapshot has not been independently rerun to reproduce those results. The original README is preserved for implementation instructions and historical attribution.

## License

[MIT License](LICENSE).
