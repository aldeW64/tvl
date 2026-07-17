import numpy as np
import torch 
import torch.nn as nn
import timm 
import open_clip
from typing import Any, Dict, Optional
from types import SimpleNamespace
from collections import OrderedDict

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    TACTILE="tactile"
)

CLIP_VISION_MODEL = "ViT-L-14"
CLIP_PRETRAIN_DATA = "datacomp_xl_s13b_b90k"

tokenizer = open_clip.get_tokenizer(CLIP_VISION_MODEL)

class TVL(nn.Module):
    def __init__(
        self, active_modalities = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT], 
        clip_vision_model=CLIP_VISION_MODEL, 
        clip_pretrain_data=CLIP_PRETRAIN_DATA, 
        tactile_model="vit_tiny_patch16_224", 
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        common_latent_dim: int = None, # for imagebind this is set to 1024, and last layer has width 1280 (ViT-H-14)
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        feature_mode: str = "pooled",
    ):
        super(TVL, self).__init__()
        assert len(active_modalities) > 1, "At least two modalities must be active"
        if feature_mode not in {"pooled", "sequence", "both"}:
            raise ValueError("feature_mode must be one of: pooled, sequence, both")
        self.active_modalities = active_modalities
        self.feature_mode = feature_mode
        self.clip, _, self.vision_preprocess = open_clip.create_model_and_transforms(clip_vision_model, pretrained=clip_pretrain_data)
        self.tokenizer = open_clip.get_tokenizer(clip_vision_model)
        
        if common_latent_dim is not None: 
            # then we will put all the modality head self.modality_head
            assert common_latent_dim > 0, "common_latent_dim must be positive"
            num_classes = 0 
        else:
            # we merge the modality head into the model
            num_classes = self.clip.transformer.width
        
        self.tactile_encoder = timm.create_model(tactile_model, pretrained=False, num_classes=num_classes, global_pool="avg", drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        modality_heads = {}
        self.common_latent_dim = common_latent_dim
        if common_latent_dim is not None:
            for modality in self.active_modalities:
                if modality == ModalityType.TACTILE:
                    modality_heads[modality] = nn.Linear(self.tactile_encoder.num_features, common_latent_dim, bias=False)
                else:
                    modality_heads[modality] = nn.Linear(self.clip.transformer.width, common_latent_dim, bias=False)
        self.modality_heads = nn.ModuleDict(modality_heads)

        # by default, we freeze openclip 
        self.freeze_vision()
        self.freeze_text()

        if ModalityType.VISION not in self.active_modalities:
            # we remove the clip.visual module
            del self.clip.visual
        if ModalityType.TEXT not in self.active_modalities:
            # we remove the clip.transformer module
            del self.clip.transformer
        
        # we clear torch cache 
        torch.cuda.empty_cache()

    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        return features / torch.norm(features, dim=-1, keepdim=True).clamp_min(1e-12)

    def freeze_openclip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    def freeze_vision(self):
        for param in self.clip.visual.parameters():
            param.requires_grad = False
    
    def freeze_tactile(self):
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False
    
    def freeze_text(self):
        for param in self.clip.transformer.parameters():
            param.requires_grad = False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(TVL, self).state_dict(destination, prefix, keep_vars)
        # we remove all clip related weights and only save the tactile encoder
        new_state_dict = OrderedDict()
        for k in state_dict:
            if "clip" not in k:
                new_state_dict[k] = state_dict[k]
        del state_dict
        return new_state_dict

    def _encode_openclip_vision_sequence(self, image: torch.Tensor) -> torch.Tensor:
        """Return OpenCLIP ViT patch tokens before final pooling/projection."""
        visual = self.clip.visual
        if not all(hasattr(visual, attr) for attr in ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")):
            raise NotImplementedError("sequence feature_mode currently supports OpenCLIP ViT visual backbones")

        x = visual.conv1(image)  # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, grid**2, width)
        cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        return x[:, 1:, :]

    def _encode_tactile_sequence(self, tactile: torch.Tensor) -> torch.Tensor:
        """Return tactile ViT patch tokens from TIMM forward_features()."""
        x = self.tactile_encoder.forward_features(tactile)
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)
        if x.ndim != 3:
            raise RuntimeError(f"Expected tactile sequence features with shape (B, N, D), got {tuple(x.shape)}")
        if getattr(self.tactile_encoder, "num_prefix_tokens", 0) > 0:
            x = x[:, self.tactile_encoder.num_prefix_tokens:, :]
        elif getattr(self.tactile_encoder, "cls_token", None) is not None and x.shape[1] > 1:
            x = x[:, 1:, :]
        return x

    def _format_feature_output(
        self,
        pooled: Optional[torch.Tensor],
        sequence: Optional[torch.Tensor],
        feature_mode: str,
    ):
        if feature_mode == "pooled":
            return pooled
        if feature_mode == "sequence":
            return sequence
        return {"pooled": pooled, "sequence": sequence}

    def forward(self, input_dict : dict, feature_mode: Optional[str] = None):
        # dictionary should have keys: vision, tactile, text
        # vision: (batch, 3, 224, 224)
        # tactile: (batch, 3, 224, 224)
        # text: (batch, 77)
        feature_mode = feature_mode or self.feature_mode
        if feature_mode not in {"pooled", "sequence", "both"}:
            raise ValueError("feature_mode must be one of: pooled, sequence, both")
        out_dict = {}
        if ModalityType.VISION in input_dict.keys():
            with torch.no_grad():
                vision_pooled = None
                vision_sequence = None
                if feature_mode in {"pooled", "both"}:
                    vision_pooled = self.clip.encode_image(input_dict[ModalityType.VISION], normalize=True)
                if feature_mode in {"sequence", "both"}:
                    vision_sequence = self._encode_openclip_vision_sequence(input_dict[ModalityType.VISION])
            out_dict[ModalityType.VISION] = self._format_feature_output(vision_pooled, vision_sequence, feature_mode)
        if ModalityType.TACTILE in input_dict.keys():
            tactile_pooled = None
            tactile_sequence = None
            if feature_mode in {"pooled", "both"}:
                tactile_pooled = self.tactile_encoder(input_dict[ModalityType.TACTILE])
                tactile_pooled = self._normalize_features(tactile_pooled)
            if feature_mode in {"sequence", "both"}:
                tactile_sequence = self._encode_tactile_sequence(input_dict[ModalityType.TACTILE])
            out_dict[ModalityType.TACTILE] = self._format_feature_output(tactile_pooled, tactile_sequence, feature_mode)
        if ModalityType.TEXT in input_dict.keys():
            if feature_mode == "sequence":
                raise NotImplementedError("Text sequence features are not implemented for TVL feature_mode='sequence'")
            with torch.no_grad():
                text_features = self.clip.encode_text(input_dict[ModalityType.TEXT], normalize=True)
            out_dict[ModalityType.TEXT] = self._format_feature_output(text_features, None, feature_mode)
        out_dict["logit_scale"] = self.logit_scale.exp()
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict
