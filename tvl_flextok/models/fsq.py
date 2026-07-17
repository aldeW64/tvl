"""Finite scalar quantization for ordered TVL-FlexTok registers."""

from math import prod
from typing import Iterable, Tuple

import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    """Project register features to independent bounded scalar code dimensions.

    The quantizer is codebook free. During training it uses a straight-through
    estimator, while ``codes`` contains the exact mixed-radix token ID used by
    the autoregressive stage.
    """

    def __init__(self, input_dim: int, levels: Iterable[int] = (8, 8, 8, 5, 5, 5)):
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        if not self.levels or any(level < 2 for level in self.levels):
            raise ValueError("FSQ levels must contain integers >= 2")
        self.code_dim = len(self.levels)
        self.vocab_size = prod(self.levels)
        self.project_in = nn.Linear(input_dim, self.code_dim)
        self.project_out = nn.Linear(self.code_dim, input_dim)
        self.register_buffer("_levels", torch.tensor(self.levels, dtype=torch.long), persistent=False)
        strides = [1]
        for level in self.levels[:-1]:
            strides.append(strides[-1] * level)
        self.register_buffer("_strides", torch.tensor(strides, dtype=torch.long), persistent=False)

    def _quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        levels = self._levels.to(device=z.device)
        half_width = (levels - 1).to(z.dtype) / 2
        bounded = torch.tanh(z) * half_width
        indices = torch.round(bounded + half_width).long()
        indices = torch.minimum(torch.maximum(indices, torch.zeros_like(indices)), levels - 1)
        quantized = (indices.to(z.dtype) - half_width) / half_width.clamp_min(0.5)
        return quantized, indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return (indices.long() * self._strides.to(indices.device)).sum(dim=-1)

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long().unsqueeze(-1)
        strides = self._strides.to(codes.device)
        levels = self._levels.to(codes.device)
        return torch.div(codes, strides, rounding_mode="floor") % levels

    def codes_to_quantized(self, codes: torch.Tensor) -> torch.Tensor:
        indices = self.codes_to_indices(codes)
        levels = self._levels.to(codes.device)
        half_width = (levels - 1).to(self.project_in.weight.dtype) / 2
        scalars = (indices.to(half_width.dtype) - half_width) / half_width.clamp_min(0.5)
        return self.project_out(scalars)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.project_in(x)
        quantized_scalars, indices = self._quantize(z)
        bounded = torch.tanh(z)
        straight_through = bounded + (quantized_scalars - bounded).detach()
        quantized = self.project_out(straight_through)
        return quantized, self.indices_to_codes(indices), indices
