"""Ordered register resampler with causal prefixes and finite scalar quantization."""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .fsq import FiniteScalarQuantizer


class RegisterTransformerLayer(nn.Module):
    """Update registers without ever modifying the frozen feature sequence."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, registers: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        q = self.self_norm(registers)
        registers = registers + self.self_attn(q, q, q, attn_mask=causal_mask, need_weights=False)[0]
        registers = registers + self.cross_attn(
            self.cross_norm(registers), self.memory_norm(memory), self.memory_norm(memory), need_weights=False
        )[0]
        return registers + self.ffn(self.ffn_norm(registers))


class RegisterTokenModule(nn.Module):
    """Compress a frozen patch sequence into ordered, FSQ-discrete registers."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        n_registers: int = 32,
        n_shared: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "uniform",
        use_token_type_embed: bool = True,
        fsq_levels=(8, 8, 8, 5, 5, 5),
    ):
        super().__init__()
        if not 0 < n_shared <= n_registers:
            raise ValueError("n_shared must be in [1, n_registers]")
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_registers = n_registers
        self.n_shared = n_shared
        self.n_private = n_registers - n_shared
        self.n_alignment_registers = n_shared
        self.nested_dropout = nested_dropout
        self.nested_dropout_mode = nested_dropout_mode
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.register_tokens = nn.Parameter(torch.empty(1, n_registers, hidden_dim))
        self.register_pos_embed = nn.Parameter(torch.empty(1, n_registers, hidden_dim))
        # Retain the argument for checkpoint/config compatibility, but use two
        # role vectors instead of a redundant per-position type tensor.
        self.use_token_type_embed = use_token_type_embed
        self.role_embed = nn.Parameter(torch.empty(2, hidden_dim)) if use_token_type_embed else None
        self.layers = nn.ModuleList([
            RegisterTransformerLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.quantizer = FiniteScalarQuantizer(hidden_dim, fsq_levels)
        self._k_keep_values = self._make_prefix_lengths(n_registers, nested_dropout_mode)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.trunc_normal_(self.register_pos_embed, std=0.02)
        if self.role_embed is not None:
            nn.init.trunc_normal_(self.role_embed, std=0.02)

    @staticmethod
    def _make_prefix_lengths(n_registers: int, mode: str):
        if mode == "uniform":
            return list(range(1, n_registers + 1))
        values, k = [], 1
        while k < n_registers:
            values.append(k)
            k *= 2
        return values + [n_registers]

    def sample_k_keep(self, batch_size: Optional[int] = None, device=None):
        """Sample uniformly over the configured prefix lengths.

        A scalar is retained for inference/backward compatibility. Training
        callers should provide ``batch_size`` so every example receives an
        independent prefix length.
        """
        if batch_size is None:
            index = torch.randint(len(self._k_keep_values), ()).item()
            return self._k_keep_values[index]
        values = torch.as_tensor(self._k_keep_values, device=device, dtype=torch.long)
        indices = torch.randint(len(values), (batch_size,), device=device)
        return values[indices]

    def _roles(self) -> torch.Tensor:
        if self.role_embed is None:
            return 0.0
        roles = torch.ones(self.n_registers, dtype=torch.long, device=self.register_tokens.device)
        roles[:self.n_shared] = 0
        return self.role_embed[roles].unsqueeze(0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        apply_nested_dropout: Optional[bool] = None,
        return_full_tokens: bool = False,
        k_keep: Optional[Union[int, torch.Tensor]] = None,
        return_dict: bool = False,
    ):
        if encoder_features.ndim != 3:
            raise ValueError(f"Expected encoder features (B,N,D), got {tuple(encoder_features.shape)}")
        batch = encoder_features.shape[0]
        memory = self.input_proj(encoder_features)
        registers = self.register_tokens.expand(batch, -1, -1) + self.register_pos_embed + self._roles()
        causal_mask = torch.triu(
            torch.full(
                (self.n_registers, self.n_registers),
                float("-inf"),
                device=registers.device,
                dtype=registers.dtype,
            ),
            diagonal=1,
        )
        for layer in self.layers:
            registers = layer(registers, memory, causal_mask)
        continuous = self.norm(registers)
        prequantized_scalars = torch.tanh(self.quantizer.project_in(continuous))
        quantized, codes, indices = self.quantizer(continuous)

        use_dropout = apply_nested_dropout if apply_nested_dropout is not None else self.nested_dropout and self.training
        if k_keep is None:
            k_keep = (
                self.sample_k_keep(batch, registers.device)
                if use_dropout else
                torch.full((batch,), self.n_registers, device=registers.device, dtype=torch.long)
            )
        elif not torch.is_tensor(k_keep):
            k_keep = torch.full((batch,), int(k_keep), device=registers.device, dtype=torch.long)
        else:
            k_keep = k_keep.to(device=registers.device, dtype=torch.long).reshape(-1)
            if k_keep.numel() == 1:
                k_keep = k_keep.expand(batch)
        if k_keep.shape != (batch,):
            raise ValueError(f"k_keep must be scalar or shape ({batch},), got {tuple(k_keep.shape)}")
        if bool(((k_keep < 1) | (k_keep > self.n_registers)).any()):
            raise ValueError(f"k_keep must be in [1,{self.n_registers}]")
        active_mask = torch.arange(self.n_registers, device=registers.device)[None, :] < k_keep[:, None]
        masked = quantized * active_mask.unsqueeze(-1)

        result: Dict[str, torch.Tensor] = {
            "continuous": continuous,
            "prequantized_scalars": prequantized_scalars,
            "quantized": quantized,
            "masked_quantized": masked,
            "codes": codes,
            "indices": indices,
            "active_mask": active_mask,
            "k_keep": k_keep,
        }
        if return_dict:
            return result

        shared = masked[:, :self.n_shared]
        private = masked[:, self.n_shared:]
        if return_full_tokens:
            return shared, private, k_keep, quantized
        return shared, private, k_keep
