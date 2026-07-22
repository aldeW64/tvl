"""Rectified-flow reconstruction of frozen VAE latents from register tokens."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    embedding = t[:, None] * frequencies[None] * 1000
    embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


def sequence_sincos(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Absolute positions for ordered register-memory tokens."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
    )
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    embedding = torch.cat([(positions * frequencies).sin(), (positions * frequencies).cos()], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1])).unsqueeze(0)


def spatial_sincos(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 4:
        raise ValueError("flow hidden_dim must be divisible by 4")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij",
    )
    omega = torch.exp(-math.log(10000) * torch.arange(dim // 4, device=device, dtype=dtype) / max(dim // 4 - 1, 1))
    return torch.cat([
        torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega),
        torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega),
    ], dim=-1).unsqueeze(0)


class FlowTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, condition: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, dim=-1)
        q = self._modulate(self.norm1(x), shift1, scale1)
        x = x + gate1[:, None] * self.self_attn(q, q, q, need_weights=False)[0]
        normalized_memory = self.memory_norm(memory)
        x = x + self.cross_attn(
            self.norm2(x), normalized_memory, normalized_memory,
            key_padding_mask=memory_padding_mask, need_weights=False,
        )[0]
        x = x + gate2[:, None] * self.ffn(self._modulate(self.norm3(x), shift2, scale2))
        return x


class LatentFlowDecoder(nn.Module):
    """Conditional velocity model over 4-channel VAE latents."""

    def __init__(
        self,
        register_dim: int = 512,
        hidden_dim: int = 512,
        depth: int = 8,
        n_heads: int = 8,
        patch_size: int = 2,
        latent_channels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        patch_dim = latent_channels * patch_size * patch_size
        self.patch_in = nn.Conv2d(latent_channels, hidden_dim, patch_size, stride=patch_size)
        self.register_proj = nn.Linear(register_dim, hidden_dim)
        self.null_register = nn.Parameter(torch.zeros(1, 1, register_dim))
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.blocks = nn.ModuleList([FlowTransformerBlock(hidden_dim, n_heads, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.patch_out = nn.Linear(hidden_dim, patch_dim)
        # A small nonzero head lets the reconstruction objective reach the
        # register tokenizer on the first optimization step.
        nn.init.normal_(self.patch_out.weight, std=0.02)
        nn.init.zeros_(self.patch_out.bias)

    def _unpatchify(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b = patches.shape[0]
        p, c = self.patch_size, self.latent_channels
        patches = patches.view(b, height, width, p, p, c)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(b, c, height * p, width * p)

    def _apply_null_condition(
        self, registers: torch.Tensor, register_padding_mask: Optional[torch.Tensor],
        drop_mask: torch.Tensor,
    ):
        if register_padding_mask is None:
            register_padding_mask = torch.zeros(
                registers.shape[:2], device=registers.device, dtype=torch.bool
            )
        if not bool(drop_mask.any()):
            return registers, register_padding_mask
        registers = registers.clone()
        register_padding_mask = register_padding_mask.clone()
        registers[drop_mask] = self.null_register.to(registers.dtype).expand(
            int(drop_mask.sum()), registers.shape[1], -1
        )
        register_padding_mask[drop_mask] = True
        register_padding_mask[drop_mask, 0] = False
        return registers, register_padding_mask

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, registers: torch.Tensor,
        register_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        patches = self.patch_in(x_t)
        h, w = patches.shape[-2:]
        x = patches.flatten(2).transpose(1, 2)
        x = x + spatial_sincos(h, w, self.hidden_dim, x.device, x.dtype)
        memory = self.register_proj(registers)
        memory = memory + sequence_sincos(
            memory.shape[1], self.hidden_dim, memory.device, memory.dtype
        )
        condition = self.time_mlp(timestep_embedding(t, self.hidden_dim))
        for block in self.blocks:
            x = block(x, memory, condition, register_padding_mask)
        return self._unpatchify(self.patch_out(self.final_norm(x)), h, w)

    def flow_loss(
        self,
        target_latents: torch.Tensor,
        registers: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        condition_dropout: float = 0.0,
        register_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            target_latents.shape, device=target_latents.device,
            dtype=target_latents.dtype, generator=generator,
        )
        t = torch.rand(
            target_latents.shape[0], device=target_latents.device,
            dtype=target_latents.dtype, generator=generator,
        )
        x_t = noise + t[:, None, None, None] * (target_latents - noise)
        target_velocity = target_latents - noise
        if condition_dropout > 0:
            drop = torch.rand(
                registers.shape[0], device=registers.device, generator=generator
            ) < condition_dropout
            registers, register_padding_mask = self._apply_null_condition(
                registers, register_padding_mask, drop
            )
        return F.mse_loss(
            self(x_t, t, registers, register_padding_mask), target_velocity
        )

    @torch.no_grad()
    def sample(
        self,
        registers: torch.Tensor,
        latent_shape,
        steps: int = 25,
        noise: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        register_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = noise if noise is not None else torch.randn(
            registers.shape[0], *latent_shape, device=registers.device, dtype=registers.dtype
        )
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((x.shape[0],), step / steps, device=x.device, dtype=x.dtype)
            conditional = self(x, t, registers, register_padding_mask)
            if guidance_scale == 1.0:
                velocity = conditional
            else:
                null_registers, null_mask = self._apply_null_condition(
                    registers, register_padding_mask,
                    torch.ones(registers.shape[0], device=registers.device, dtype=torch.bool),
                )
                unconditional = self(x, t, null_registers, null_mask)
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            x = x + dt * velocity
        return x


class FrozenVAECodec(nn.Module):
    """Lazy Diffusers AutoencoderKL wrapper used as a fixed latent codec."""

    def __init__(
        self,
        model_id: str = "EPFL-VILAB/flextok_vae_c4",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from diffusers.models import AutoencoderKL
        except ImportError as exc:
            raise ImportError("diffusers is required for FlexTok VAE reconstruction") from exc
        self.vae = AutoencoderKL.from_pretrained(
            model_id, cache_dir=cache_dir, low_cpu_mem_usage=False,
            local_files_only=local_files_only,
        ).eval()
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))
        for parameter in self.vae.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor, sample: bool = False) -> torch.Tensor:
        posterior = self.vae.encode(pixels).latent_dist
        latents = posterior.sample() if sample else posterior.mode()
        return latents * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents / self.scaling_factor).sample
