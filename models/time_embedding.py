"""
Timestep embedding module for CFlow-LM.

Provides sinusoidal + MLP time embeddings that inject the continuous
diffusion timestep t ∈ [0,1] into the transformer hidden states.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for continuous time t ∈ [0,1].

    Maps scalar t to a rich frequency embedding, then projects through
    an MLP to match the transformer hidden dimension.

    Args:
        hidden_dim: Target embedding dimension (must match transformer d_model).
        max_period: Controls the minimum sinusoidal frequency.
        learnable_proj: If True, add a 2-layer MLP on top of the sinusoidal base.
    """

    def __init__(self, hidden_dim: int, max_period: float = 10000.0, learnable_proj: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_period = max_period

        # Base sinusoidal dim is hidden_dim // 2 * 2 (keeps it even)
        self.base_dim = hidden_dim

        if learnable_proj:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.SiLU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps, shape (B,) or (B, 1), values in [0, 1].
        Returns:
            emb: Shape (B, hidden_dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        t = t.float()

        half = self.base_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1 if half > 1 else 1)
        )  # (half,)

        args = t * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, base_dim)

        # Pad if hidden_dim is odd
        if self.base_dim < self.hidden_dim:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], self.hidden_dim - self.base_dim, device=emb.device)], dim=-1)

        return self.proj(emb)  # (B, hidden_dim)


class AdaptiveLayerNorm(nn.Module):
    """AdaLN: modulates LayerNorm scale/shift conditioned on time embedding.

    Used as an alternative to additive time injection when the backbone
    supports it (e.g. custom GPT-2 variants).

    Args:
        hidden_dim: Hidden dimension of the transformer.
        time_dim: Dimension of the time embedding.
    """

    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(time_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states (B, L, hidden_dim).
            t_emb: Time embedding (B, time_dim).
        Returns:
            Modulated hidden states (B, L, hidden_dim).
        """
        scale_shift = self.proj(t_emb).unsqueeze(1)  # (B, 1, 2*hidden_dim)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift
