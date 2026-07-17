"""Causal prediction of discrete FSQ register sequences."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveDecoder(nn.Module):
    """Teacher-forced Transformer that predicts target-modality register codes.

    ``source_registers`` condition the model through cross-attention. Target
    code IDs are shifted right with a BOS token, and an EOS target is appended.
    """

    def __init__(
        self,
        vocab_size: int = 64000,
        register_dim: int = 512,
        hidden_dim: int = 512,
        max_registers: int = 256,
        depth: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = vocab_size
        self.bos_id = vocab_size + 1
        self.max_registers = max_registers
        self.code_embed = nn.Embedding(vocab_size + 2, hidden_dim)
        self.position_embed = nn.Parameter(torch.empty(1, max_registers + 1, hidden_dim))
        self.modality_embed = nn.Embedding(2, hidden_dim)
        self.source_proj = nn.Linear(register_dim, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth, norm=nn.LayerNorm(hidden_dim))
        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def _decode(self, source_registers: torch.Tensor, input_ids: torch.Tensor, target_modality: torch.Tensor):
        if input_ids.shape[1] > self.max_registers + 1:
            raise ValueError("Target sequence exceeds max_registers")
        target = self.code_embed(input_ids) + self.position_embed[:, :input_ids.shape[1]]
        target = target + self.modality_embed(target_modality).unsqueeze(1)
        memory = self.source_proj(source_registers)
        length = input_ids.shape[1]
        causal = torch.triu(torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1)
        decoded = self.decoder(target, memory, tgt_mask=causal)
        return F.linear(decoded, self.code_embed.weight[:self.vocab_size + 1])

    def forward(
        self,
        source_registers: torch.Tensor,
        target_codes: torch.Tensor,
        target_modality: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = target_codes.shape[0]
        bos = torch.full((batch, 1), self.bos_id, device=target_codes.device, dtype=torch.long)
        inputs = torch.cat([bos, target_codes], dim=1)
        eos = torch.full((batch, 1), self.eos_id, device=target_codes.device, dtype=torch.long)
        targets = torch.cat([target_codes, eos], dim=1)
        return self._decode(source_registers, inputs, target_modality), targets

    def loss(self, source_registers, target_codes, target_modality) -> torch.Tensor:
        logits, targets = self(source_registers, target_codes, target_modality)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.no_grad()
    def generate(
        self,
        source_registers: torch.Tensor,
        target_modality: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        max_length = max_length or self.max_registers
        ids = torch.full(
            (source_registers.shape[0], 1), self.bos_id, device=source_registers.device, dtype=torch.long
        )
        finished = torch.zeros(source_registers.shape[0], device=source_registers.device, dtype=torch.bool)
        generated = []
        for _ in range(max_length):
            logits = self._decode(source_registers, ids, target_modality)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                values, _ = logits.topk(min(top_k, logits.shape[-1]))
                logits[logits < values[:, -1:]] = float("-inf")
            next_id = torch.distributions.Categorical(logits=logits).sample()
            finished |= next_id.eq(self.eos_id)
            generated.append(next_id.clamp_max(self.vocab_size - 1))
            ids = torch.cat([ids, next_id[:, None]], dim=1)
            if finished.all():
                break
        return torch.stack(generated, dim=1) if generated else ids[:, :0]
