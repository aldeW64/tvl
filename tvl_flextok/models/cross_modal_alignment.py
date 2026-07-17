"""Cross-modal FSQ register tokenizer over frozen TVL patch features."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from types import SimpleNamespace

from .register_tokens import RegisterTokenModule

# Re-use the modality type definitions from TVL
ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    TACTILE="tactile",
)


class CrossModalAlignmentModel(nn.Module):
    """
    TVL-FlexTok model that takes frozen encoder features and produces
    aligned cross-modal representations via register tokens.

    Args:
        modality_configs: Dict mapping modality name to encoder output config.
            Each entry: {"input_dim": int, "feature_type": "pooled" | "sequence"}
        hidden_dim: Shared hidden dimension for all register token modules.
        n_registers: Number of register tokens per modality.
        n_shared: Number of shared (aligned) registers per modality.
        n_layers: Transformer depth in each register module.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        nested_dropout: Whether to use nested dropout.
        nested_dropout_mode: "power_of_two" or "uniform".
        init_logit_scale: Initial value for learnable logit scale (temperature).
    """

    def __init__(
        self,
        modality_configs: Optional[Dict] = None,
        hidden_dim: int = 512,
        n_registers: int = 32,
        n_shared: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        nested_dropout: bool = True,
        nested_dropout_mode: str = "power_of_two",
        init_logit_scale: float = np.log(1 / 0.07),
        use_token_type_embed: bool = True,
        fsq_levels=(8, 8, 8, 5, 5, 5),
        tokenizer_input: str = "tvl",
    ):
        super().__init__()

        if tokenizer_input not in {"tvl", "vae", "vae_tvl"}:
            raise ValueError("tokenizer_input must be one of: tvl, vae, vae_tvl")

        if modality_configs is None:
            # Default: vision (OpenCLIP ViT-L-14 = 768d) and tactile (ViT = 768d)
            modality_configs = {
                ModalityType.VISION: {"input_dim": 768, "feature_type": "pooled"},
                ModalityType.TACTILE: {"input_dim": 768, "feature_type": "pooled"},
            }

        self.modality_names = list(modality_configs.keys())
        self.hidden_dim = hidden_dim
        self.n_registers = n_registers
        self.n_shared = n_shared
        self.tokenizer_input = tokenizer_input

        # Create a RegisterTokenModule for each modality
        self.register_modules = nn.ModuleDict()
        self.latent_projectors = nn.ModuleDict()
        self.memory_type_embeddings = nn.ParameterDict()
        for mod_name, config in modality_configs.items():
            self.register_modules[mod_name] = RegisterTokenModule(
                input_dim=config["input_dim"],
                hidden_dim=hidden_dim,
                n_registers=n_registers,
                n_shared=n_shared,
                n_layers=n_layers,
                n_heads=n_heads,
                dropout=dropout,
                nested_dropout=nested_dropout,
                nested_dropout_mode=nested_dropout_mode,
                use_token_type_embed=use_token_type_embed,
                fsq_levels=fsq_levels,
            )
            if tokenizer_input in {"vae", "vae_tvl"}:
                latent_channels = int(config.get("latent_channels", 4))
                latent_patch_size = int(config.get("latent_patch_size", 2))
                self.latent_projectors[mod_name] = nn.Conv2d(
                    latent_channels, config["input_dim"], latent_patch_size, stride=latent_patch_size
                )
            if tokenizer_input == "vae_tvl":
                embedding = nn.Parameter(torch.empty(2, config["input_dim"]))
                nn.init.trunc_normal_(embedding, std=0.02)
                self.memory_type_embeddings[mod_name] = embedding

        self.feature_types = {k: v["feature_type"] for k, v in modality_configs.items()}

        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        # Projection heads: map shared tokens to a normalized embedding for contrastive loss.
        # We pool the shared tokens (mean pool) then project.
        self.shared_projectors = nn.ModuleDict()
        for mod_name in modality_configs:
            self.shared_projectors[mod_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def _prepare_features(self, features: torch.Tensor, feature_type: str) -> torch.Tensor:
        """Ensure features are (B, N, D) for the register module."""
        if isinstance(features, dict):
            features = features[feature_type]
        if feature_type == "pooled" and features.ndim == 2:
            return features.unsqueeze(1)  # (B, D) -> (B, 1, D)
        if features.ndim != 3:
            raise ValueError(
                f"Expected {feature_type} features to have shape (B, N, D), got {tuple(features.shape)}"
            )
        return features

    @staticmethod
    def _latent_position(height: int, width: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
        quarter = max(dim // 4, 1)
        y, x = torch.meshgrid(
            torch.arange(height, device=reference.device, dtype=reference.dtype),
            torch.arange(width, device=reference.device, dtype=reference.dtype), indexing="ij",
        )
        omega = torch.exp(
            -math.log(10000) * torch.arange(quarter, device=reference.device, dtype=reference.dtype)
            / max(quarter - 1, 1)
        )
        position = torch.cat([
            torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega),
            torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega),
        ], dim=-1)
        if position.shape[-1] < dim:
            position = F.pad(position, (0, dim - position.shape[-1]))
        return position[:, :dim].unsqueeze(0)

    def forward(
        self,
        feature_dict: Dict[str, torch.Tensor],
        apply_nested_dropout: Optional[bool] = None,
        latent_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feature_dict: Maps modality name -> frozen encoder features.
                Vision: (B, 768) or (B, N, 768) depending on pooling.
                Tactile: (B, 768) or (B, N, 768).

        Returns:
            Dictionary with:
                - "{mod}_shared": (B, hidden_dim) L2-normalized shared embedding
                - "{mod}_private": (B, n_private, hidden_dim) private tokens
                - "{mod}_shared_tokens": (B, n_shared, hidden_dim) raw shared tokens
                - "{mod}_k_keep": int, tokens kept after nested dropout
                - "logit_scale": scalar temperature
        """
        output = {}
        active_modalities = [name for name in self.modality_names if name in feature_dict]
        use_dropout = (
            apply_nested_dropout
            if apply_nested_dropout is not None
            else self.training and any(self.register_modules[name].nested_dropout for name in active_modalities)
        )
        # Paired modalities must expose the same amount of information. Sampling
        # independently makes the contrastive target depend on two different
        # bottleneck capacities.
        k_keep = None
        if active_modalities:
            k_keep = self.register_modules[active_modalities[0]].sample_k_keep() if use_dropout else self.n_registers

        for mod_name in self.modality_names:
            if mod_name not in feature_dict:
                continue

            features = self._prepare_features(feature_dict[mod_name], self.feature_types[mod_name])
            memory_parts = []
            if self.tokenizer_input in {"tvl", "vae_tvl"}:
                semantic = features
                if self.tokenizer_input == "vae_tvl":
                    semantic = semantic + self.memory_type_embeddings[mod_name][0].view(1, 1, -1)
                memory_parts.append(semantic)
            if self.tokenizer_input in {"vae", "vae_tvl"}:
                if latent_dict is None or mod_name not in latent_dict:
                    raise ValueError(f"VAE latents are required for tokenizer_input={self.tokenizer_input}")
                latent = self.latent_projectors[mod_name](latent_dict[mod_name])
                height, width = latent.shape[-2:]
                latent = latent.flatten(2).transpose(1, 2)
                latent = latent + self._latent_position(height, width, latent.shape[-1], latent)
                if self.tokenizer_input == "vae_tvl":
                    latent = latent + self.memory_type_embeddings[mod_name][1].view(1, 1, -1)
                memory_parts.append(latent)
            features = torch.cat(memory_parts, dim=1)

            tokenized = self.register_modules[mod_name](
                features,
                apply_nested_dropout=apply_nested_dropout,
                k_keep=k_keep,
                return_dict=True,
            )
            all_tokens = tokenized["masked_quantized"]
            full_tokens = tokenized["quantized"]
            shared_tokens = all_tokens[:, :self.n_shared]
            continuous_shared_tokens = tokenized["continuous"][:, :self.n_shared]
            private_tokens = all_tokens[:, self.n_shared:]

            # Pool only non-zeroed shared tokens to avoid dilution from nested dropout.
            # k_keep indicates how many shared tokens are active (rest are zeroed).
            if k_keep < shared_tokens.shape[1]:
                shared_pooled = shared_tokens[:, :k_keep, :].mean(dim=1)
            else:
                shared_pooled = shared_tokens.mean(dim=1)  # (B, hidden_dim)
            shared_proj = self.shared_projectors[mod_name](shared_pooled)
            shared_proj = F.normalize(shared_proj, dim=-1)
            continuous_shared_pooled = continuous_shared_tokens[:, :min(k_keep, self.n_shared)].mean(dim=1)
            continuous_shared_proj = F.normalize(
                self.shared_projectors[mod_name](continuous_shared_pooled), dim=-1
            )

            output[f"{mod_name}_shared"] = shared_proj
            output[f"{mod_name}_continuous_shared"] = continuous_shared_proj
            output[f"{mod_name}_private"] = private_tokens
            output[f"{mod_name}_shared_tokens"] = shared_tokens
            output[f"{mod_name}_k_keep"] = k_keep
            # All register tokens (for reconstruction decoder)
            output[f"{mod_name}_all_tokens"] = all_tokens
            output[f"{mod_name}_all_tokens_full"] = full_tokens
            output[f"{mod_name}_continuous_tokens"] = tokenized["continuous"]
            output[f"{mod_name}_prequantized_scalars"] = tokenized["prequantized_scalars"]
            output[f"{mod_name}_code_ids"] = tokenized["codes"]
            output[f"{mod_name}_active_mask"] = tokenized["active_mask"]

        output["logit_scale"] = self.logit_scale.exp()
        output["k_keep"] = k_keep
        return output



class FlexTokWrapper(nn.Module):
    """
    Full TVL-FlexTok pipeline: frozen encoders + cross-modal alignment.

    This wraps the frozen TVL model and the trainable register tokenizer into
    a single forward pass.
    """

    def __init__(
        self,
        frozen_encoder,
        alignment_model: CrossModalAlignmentModel,
        feature_mode: str = "sequence",
    ):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.alignment_model = alignment_model
        self.feature_mode = feature_mode

        # Freeze the encoder
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run frozen Stage-1 encoders."""
        self.frozen_encoder.eval()
        out = self.frozen_encoder(input_dict, feature_mode=self.feature_mode)
        # Remove non-feature keys
        out.pop("logit_scale", None)
        out.pop("logit_bias", None)
        return out

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward: frozen encode -> alignment."""
        frozen_features = self.encode(input_dict)
        return self.alignment_model(frozen_features)


# Checkpoint/import compatibility for earlier experiments.
Stage2Wrapper = FlexTokWrapper
