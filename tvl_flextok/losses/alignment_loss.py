"""
Cross-Modal Alignment Loss for TVL-FlexTok.

Combines:
  1. Contrastive loss on shared register tokens (CLIP-style) to align
     overlapping information across modalities.
  2. Tokenizer diagnostics that monitor register variance/utilization.

The contrastive loss operates on the pooled+projected shared embeddings.
Feature reconstruction is intentionally not optimized here: TVL-FlexTok
registers are a compact bottleneck and should be free to discard irrelevant
frozen-encoder details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from itertools import combinations
from typing import Dict, Optional, List

def topk_accuracy(output, target, topk=(1,)):
    """Top-k accuracy (matches timm.utils.accuracy interface)."""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class CrossModalAlignmentLoss(nn.Module):
    """
    Loss for TVL-FlexTok cross-modal alignment.

    Components:
        1. contrastive_loss: Symmetric CLIP-style contrastive loss between
           shared embeddings of different modalities.
    Args:
        modality_names: List of modality names to align.
        contrastive_weight: Weight for contrastive alignment loss.
        temperature_clamp: Max/min for learned temperature to prevent instability.
    """

    def __init__(
        self,
        modality_names: List[str] = None,
        contrastive_weight: float = 1.0,
        continuous_contrastive_weight: float = 0.25,
        diversity_weight: float = 0.1,
        diversity_min_std: float = 0.2,
        temperature_clamp: float = 100.0,
    ):
        super().__init__()
        if modality_names is None:
            modality_names = ["vision", "tactile"]
        self.modality_names = modality_names
        self.contrastive_weight = contrastive_weight
        self.continuous_contrastive_weight = continuous_contrastive_weight
        self.diversity_weight = diversity_weight
        self.diversity_min_std = diversity_min_std
        self.temperature_clamp = temperature_clamp

    def contrastive_loss(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> tuple:
        """
        Symmetric CLIP-style contrastive loss.

        Args:
            feat_a: (B, D) L2-normalized shared embedding for modality A.
            feat_b: (B, D) L2-normalized shared embedding for modality B.
            logit_scale: Scalar temperature (exp of learned parameter).

        Returns:
            loss: Scalar contrastive loss.
            affinity_matrix: (B, B) similarity matrix for metrics.
        """
        # Clamp logit scale for stability
        logit_scale = torch.clamp(logit_scale, max=self.temperature_clamp)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.nn.functional import all_gather
            all_a = torch.cat(all_gather(feat_a), dim=0)
            all_b = torch.cat(all_gather(feat_b), dim=0)
            offset = dist.get_rank() * feat_a.shape[0]
            labels = torch.arange(feat_a.shape[0], device=feat_a.device, dtype=torch.long) + offset
            affinity_ab = logit_scale * feat_a @ all_b.T
            affinity_ba = logit_scale * feat_b @ all_a.T
            loss_ab = F.cross_entropy(affinity_ab, labels)
            loss_ba = F.cross_entropy(affinity_ba, labels)
            # Retrieval metrics remain rank-local; the optimized negatives are global.
            return (loss_ab + loss_ba) / 2, logit_scale * feat_a @ feat_b.T
        labels = torch.arange(feat_a.shape[0], device=feat_a.device, dtype=torch.long)
        affinity = logit_scale * feat_a @ feat_b.T
        loss_ab = F.cross_entropy(affinity, labels)
        loss_ba = F.cross_entropy(affinity.T, labels)
        return (loss_ab + loss_ba) / 2, affinity

    def get_acc_from_affinity(self, affinity_matrix: torch.Tensor) -> tuple:
        """Compute top-1 and top-5 retrieval accuracy from affinity matrix."""
        labels = torch.arange(affinity_matrix.shape[0], device=affinity_matrix.device, dtype=torch.long)
        acc1, acc5 = topk_accuracy(affinity_matrix, labels, topk=(1, min(5, affinity_matrix.shape[0])))
        acc1_t, acc5_t = topk_accuracy(affinity_matrix.T, labels, topk=(1, min(5, affinity_matrix.shape[0])))
        return (acc1 + acc1_t) / 2, (acc5 + acc5_t) / 2

    def forward(
        self,
        alignment_output: Dict[str, torch.Tensor],
        frozen_features: Optional[Dict[str, torch.Tensor]] = None,
        output_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined alignment loss.

        Args:
            alignment_output: Output from CrossModalAlignmentModel.forward().
            frozen_features: Deprecated compatibility argument; ignored.
            output_dict: If True, return dict of all losses and metrics.

        Returns:
            Dict with losses and accuracy metrics, or scalar average loss.
        """
        logit_scale = alignment_output["logit_scale"]
        losses = {}
        total_contrastive = 0.0
        n_pairs = 0

        # Contrastive loss for all modality pairs
        active_modalities = [m for m in self.modality_names if f"{m}_shared" in alignment_output]
        pairs = list(combinations(active_modalities, 2))

        for mod_a, mod_b in pairs:
            feat_a = alignment_output[f"{mod_a}_shared"]
            feat_b = alignment_output[f"{mod_b}_shared"]
            loss, affinity = self.contrastive_loss(feat_a, feat_b, logit_scale)
            acc1, acc5 = self.get_acc_from_affinity(affinity)

            pair_name = f"{mod_a}_{mod_b}"
            losses[f"contrastive_{pair_name}"] = loss
            losses[f"acc1_{pair_name}"] = acc1
            losses[f"acc5_{pair_name}"] = acc5
            total_contrastive += loss
            n_pairs += 1

        if n_pairs > 0:
            losses["contrastive_avg"] = total_contrastive / n_pairs

        continuous_losses = []
        for mod_a, mod_b in pairs:
            feat_a = alignment_output.get(f"{mod_a}_continuous_shared")
            feat_b = alignment_output.get(f"{mod_b}_continuous_shared")
            if feat_a is None or feat_b is None:
                continue
            loss, _ = self.contrastive_loss(feat_a, feat_b, logit_scale)
            losses[f"continuous_contrastive_{mod_a}_{mod_b}"] = loss
            continuous_losses.append(loss)
        if continuous_losses:
            losses["continuous_contrastive_avg"] = torch.stack(continuous_losses).mean()

        # Tokenizer-quality diagnostics. These are not optimized directly,
        # but they catch collapsed or unused registers during training.
        for mod_name in active_modalities:
            all_tokens = alignment_output.get(f"{mod_name}_all_tokens_full")
            if all_tokens is None:
                continue
            per_register_std = all_tokens.float().std(dim=0, correction=0).mean(dim=-1)
            losses[f"token_variance_{mod_name}"] = per_register_std.mean()
            losses[f"token_utilization_{mod_name}"] = (per_register_std > 1e-3).float().mean()
            code_ids = alignment_output.get(f"{mod_name}_code_ids")
            if code_ids is not None:
                losses[f"fsq_code_utilization_{mod_name}"] = (
                    torch.unique(code_ids).numel() / max(code_ids.numel(), 1)
                )

            scalars = alignment_output.get(f"{mod_name}_prequantized_scalars")
            if scalars is not None:
                sample_std = scalars.float().std(dim=0, correction=0)
                register_std = scalars.float().std(dim=1, correction=0)
                losses[f"fsq_sample_std_{mod_name}"] = sample_std.mean()
                losses[f"fsq_register_std_{mod_name}"] = register_std.mean()
                losses[f"diversity_{mod_name}"] = 0.5 * (
                    F.relu(self.diversity_min_std - sample_std).mean()
                    + F.relu(self.diversity_min_std - register_std).mean()
                )

        # Combined loss
        diversity_losses = [value for key, value in losses.items() if key.startswith("diversity_")]
        diversity = torch.stack(diversity_losses).mean() if diversity_losses else 0.0
        losses["diversity"] = diversity
        total = (
            self.contrastive_weight * losses.get("contrastive_avg", 0.0)
            + self.continuous_contrastive_weight * losses.get("continuous_contrastive_avg", 0.0)
            + self.diversity_weight * diversity
        )
        losses["total_loss"] = total

        # Average accuracy across pairs
        acc1_vals = [v for k, v in losses.items() if k.startswith("acc1_")]
        acc5_vals = [v for k, v in losses.items() if k.startswith("acc5_")]
        if acc1_vals:
            losses["acc1_avg"] = torch.stack(acc1_vals).mean()
            losses["acc5_avg"] = torch.stack(acc5_vals).mean()

        losses["logit_scale"] = logit_scale

        if output_dict:
            return losses
        return losses["total_loss"]
