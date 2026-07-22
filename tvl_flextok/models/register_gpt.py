"""Text-conditioned autoregressive model over discrete FSQ register IDs."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionedRegisterGPT(nn.Module):
    """Predict ordered FlexTok codes from text with a shared multimodal GPT."""

    def __init__(
        self,
        vocab_size: int,
        text_dim: int,
        hidden_dim: int = 512,
        max_registers: int = 64,
        depth: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_modalities: int = 2,
    ):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_registers = max_registers
        self.code_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.modality_bos = nn.Embedding(num_modalities, hidden_dim)
        self.position = nn.Parameter(torch.empty(1, max_registers, hidden_dim))
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.null_text = nn.Parameter(torch.empty(1, 1, hidden_dim))
        layer = nn.TransformerDecoderLayer(
            hidden_dim, n_heads, hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output.weight = self.code_embedding.weight
        nn.init.trunc_normal_(self.code_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.modality_bos.weight, std=0.02)
        nn.init.trunc_normal_(self.null_text, std=0.02)

    def _memory(
        self,
        text_features: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor],
        condition_drop_mask: Optional[torch.Tensor],
    ):
        memory = self.text_projection(text_features)
        if text_padding_mask is None:
            text_padding_mask = torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
        if condition_drop_mask is not None and bool(condition_drop_mask.any()):
            memory = memory.clone()
            text_padding_mask = text_padding_mask.clone()
            memory[condition_drop_mask] = self.null_text.to(memory.dtype).expand(
                int(condition_drop_mask.sum()), memory.shape[1], -1
            )
            text_padding_mask[condition_drop_mask] = True
            text_padding_mask[condition_drop_mask, 0] = False
        return memory, text_padding_mask

    def forward(
        self,
        text_features: torch.Tensor,
        target_codes: torch.Tensor,
        modality_ids: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        condition_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_codes.ndim != 2 or not 1 <= target_codes.shape[1] <= self.max_registers:
            raise ValueError(f"target_codes must have shape (B,1..{self.max_registers})")
        batch, length = target_codes.shape
        bos = self.modality_bos(modality_ids).unsqueeze(1)
        previous = self.code_embedding(target_codes[:, :-1])
        target = torch.cat([bos, previous], dim=1) + self.position[:, :length]
        memory, text_padding_mask = self._memory(
            text_features, text_padding_mask, condition_drop_mask
        )
        causal = torch.triu(
            torch.ones(length, length, device=target.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.decoder(
            target, memory, tgt_mask=causal,
            memory_key_padding_mask=text_padding_mask,
        )
        return self.output(hidden)

    def loss(self, *args, **kwargs):
        target_codes = args[1] if len(args) > 1 else kwargs["target_codes"]
        logits = self(*args, **kwargs)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_codes.reshape(-1))
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == target_codes).float().mean()
        first_accuracy = (predictions[:, 0] == target_codes[:, 0]).float().mean()
        return loss, accuracy, first_accuracy

    @torch.no_grad()
    def generate(
        self,
        text_features: torch.Tensor,
        modality_ids: torch.Tensor,
        text_padding_mask: Optional[torch.Tensor] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        guidance_scale: float = 1.0,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        max_tokens = max_tokens or self.max_registers
        if not 1 <= max_tokens <= self.max_registers:
            raise ValueError(f"max_tokens must be in [1,{self.max_registers}]")
        codes = torch.empty(
            text_features.shape[0], 0, device=text_features.device, dtype=torch.long
        )
        null_drop = torch.ones(text_features.shape[0], device=text_features.device, dtype=torch.bool)
        for _ in range(max_tokens):
            # forward() shifts target_codes right. The final placeholder is
            # ignored as an input and reserves the position being predicted.
            decoder_input = torch.cat([
                codes,
                torch.zeros(codes.shape[0], 1, device=codes.device, dtype=codes.dtype),
            ], dim=1)
            conditional = self(
                text_features, decoder_input, modality_ids, text_padding_mask
            )[:, -1]
            if guidance_scale != 1.0:
                unconditional = self(
                    text_features, decoder_input, modality_ids,
                    text_padding_mask, null_drop,
                )[:, -1]
                logits = unconditional + guidance_scale * (conditional - unconditional)
            else:
                logits = conditional
            if not sample or temperature <= 0:
                next_code = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if 0 < top_k < logits.shape[-1]:
                    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                next_code = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            codes = torch.cat([codes, next_code[:, None]], dim=1)
        return codes
