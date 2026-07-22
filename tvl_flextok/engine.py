"""Training engine for the discrete register tokenizer and its generative heads."""

import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tvl_enc.tacvis import RGB_MEAN, RGB_STD, TAC_MEAN, TAC_STD
from tvl_enc.tvl import ModalityType
from tvl_enc.util import lr_sched, misc
from tvl_flextok.losses.alignment_loss import CrossModalAlignmentLoss
from tvl_flextok.models.flow_decoder import FrozenVAECodec, LatentFlowDecoder
from tvl_flextok.models.register_gpt import TextConditionedRegisterGPT


MODALITIES = (ModalityType.VISION, ModalityType.TACTILE)
MODALITY_IDS = {ModalityType.VISION: 0, ModalityType.TACTILE: 1}


def _move_samples(samples, device):
    result = {}
    for key, value in samples.items():
        if isinstance(value, list):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            continue
        value = value.to(device, non_blocking=True)
        if value.ndim > 4:
            value = value.squeeze(0)
        result[key] = value
    return result


def _pixels_for_codec(tensor: torch.Tensor, modality: str) -> torch.Tensor:
    mean, std = (RGB_MEAN, RGB_STD) if modality == ModalityType.VISION else (TAC_MEAN, TAC_STD)
    mean = tensor.new_tensor(mean).view(1, 3, 1, 1)
    std = tensor.new_tensor(std).view(1, 3, 1, 1)
    return ((tensor * std + mean).clamp(0, 1) * 2 - 1)


def _distributed_mean(value: float, device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    return tensor.item()


class FlexTokTrainingSystem(nn.Module):
    def __init__(self, tokenizer, codec, flow_decoders=None, register_gpt=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.codec = codec
        self.flow_decoders = flow_decoders
        self.register_gpt = register_gpt

    def encode_latents(self, samples, sample=False):
        return {
            modality: self.codec.encode(_pixels_for_codec(samples[modality], modality), sample=sample)
            for modality in MODALITIES
        }

    def encode_registers(self, samples, nested_dropout=None, latents=None):
        modality_samples = {key: samples[key] for key in MODALITIES}
        with torch.no_grad():
            frozen = self.tokenizer.encode(modality_samples)
            if latents is None and self.tokenizer.alignment_model.tokenizer_input in {"vae", "vae_tvl"}:
                latents = self.encode_latents(samples, sample=False)
        return self.tokenizer.alignment_model(
            frozen, apply_nested_dropout=nested_dropout, latent_dict=latents
        )

    def forward(self, samples, contrastive_loss=None, stage="alignment"):
        if stage == "alignment":
            return self.alignment_losses(samples, contrastive_loss)
        if stage == "generation":
            return self.generation_losses(samples)
        raise ValueError(f"Unknown training stage: {stage}")

    def alignment_losses(self, samples, contrastive_loss):
        with torch.no_grad():
            targets = self.encode_latents(samples, sample=self.training)
        output = self.encode_registers(samples, latents=targets)
        align = contrastive_loss(output, output_dict=True)
        losses = {
            "contrastive": align.get("contrastive_avg", output["logit_scale"] * 0),
            "continuous_contrastive": align.get("continuous_contrastive_avg", output["logit_scale"] * 0),
            "diversity": align.get("diversity", output["logit_scale"] * 0),
            "acc1": align.get("acc1_avg", output["logit_scale"] * 0),
        }
        for key, value in align.items():
            if key.startswith((
                "token_variance_", "token_utilization_", "fsq_code_utilization_",
                "fsq_sample_std_", "fsq_register_std_",
            )):
                losses[key] = value
        total_flow = output["logit_scale"] * 0
        for modality in MODALITIES:
            active_registers = output[f"{modality}_all_tokens"]
            register_padding_mask = ~output[f"{modality}_active_mask"]
            generator = None
            if not self.training:
                generator = torch.Generator(device=targets[modality].device).manual_seed(
                    1701 + MODALITY_IDS[modality]
                )
            flow = self.flow_decoders[modality].flow_loss(
                targets[modality], active_registers, generator=generator,
                condition_dropout=self.flow_condition_dropout if self.training else 0.0,
                register_padding_mask=register_padding_mask,
            )
            losses[f"flow_{modality}"] = flow
            total_flow = total_flow + flow
        losses["flow"] = total_flow / len(MODALITIES)
        for modality in MODALITIES:
            codes = output[f"{modality}_code_ids"][output[f"{modality}_active_mask"]]
            losses[f"code_utilization_{modality}"] = codes.unique().numel() / max(codes.numel(), 1)
        # Keep every ordered-register parameter in the DDP graph even when a
        # sampled short prefix correctly gives the suffix zero gradient.
        graph_anchor = sum(output[f"{modality}_continuous_tokens"].sum() for modality in MODALITIES) * 0
        return align["total_loss"] + self.reconstruction_weight * losses["flow"] + graph_anchor, losses, output

    @torch.no_grad()
    def encode_text(self, samples):
        text = samples[ModalityType.TEXT]
        if text.ndim == 3 and text.shape[1] == 1:
            text = text[:, 0]
        features = self.tokenizer.encode_text(text)
        return features, text.eq(0)

    def generation_losses(self, samples):
        with torch.no_grad():
            latents = self.encode_latents(samples, sample=False)
            output = self.encode_registers(samples, nested_dropout=False, latents=latents)
            text_features, text_padding_mask = self.encode_text(samples)
        losses = {}
        ce_values = []
        for modality in MODALITIES:
            codes = output[f"{modality}_code_ids"]
            modality_ids = torch.full(
                (codes.shape[0],), MODALITY_IDS[modality],
                device=codes.device, dtype=torch.long,
            )
            drop = None
            if self.training and self.text_condition_dropout > 0:
                drop = torch.rand(codes.shape[0], device=codes.device) < self.text_condition_dropout
            ce, accuracy, first_accuracy = self.register_gpt.loss(
                text_features, codes, modality_ids, text_padding_mask, drop
            )
            losses[f"ce_{modality}"] = ce
            losses[f"token_accuracy_{modality}"] = accuracy
            losses[f"first_token_accuracy_{modality}"] = first_accuracy
            ce_values.append(ce)
            if not self.training:
                shuffled = text_features.roll(1, dims=0)
                shuffled_mask = text_padding_mask.roll(1, dims=0)
                shuffled_ce, _, shuffled_first = self.register_gpt.loss(
                    shuffled, codes, modality_ids, shuffled_mask
                )
                losses[f"shuffled_text_ce_{modality}"] = shuffled_ce
                losses[f"text_ce_gap_{modality}"] = shuffled_ce - ce
                losses[f"shuffled_first_accuracy_{modality}"] = shuffled_first
        losses["generation"] = sum(ce_values) / len(ce_values)
        losses["perplexity"] = losses["generation"].detach().clamp(max=20).exp()
        return losses["generation"], losses, output


def load_alignment_stack(system, checkpoint):
    """Restore alignment weights, permitting only the pre-v4 null-register omission."""
    system.tokenizer.alignment_model.load_state_dict(
        checkpoint["alignment_model"], strict=True
    )
    if checkpoint.get("format_version", 1) >= 4:
        system.flow_decoders.load_state_dict(checkpoint["flow_decoders"], strict=True)
        return

    result = system.flow_decoders.load_state_dict(
        checkpoint["flow_decoders"], strict=False
    )
    allowed_missing = {f"{modality}.null_register" for modality in MODALITIES}
    unexpected = set(result.unexpected_keys)
    disallowed_missing = set(result.missing_keys) - allowed_missing
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Legacy alignment checkpoint is incompatible: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}"
        )
    if result.missing_keys:
        print(
            "Loaded pre-v4 alignment flow weights; initialized missing learned "
            f"null registers to zero: {sorted(result.missing_keys)}"
        )


def _build_system(args, build_model: Callable, device):
    tokenizer = build_model(args, device)
    cache_dir = args.codec_cache_dir or os.environ.get("HF_HOME")
    codec = FrozenVAECodec(
        args.codec_id, cache_dir=cache_dir,
        local_files_only=args.codec_local_files_only,
    ).to(device)
    flow_decoders = nn.ModuleDict({
        modality: LatentFlowDecoder(
            register_dim=args.hidden_dim,
            hidden_dim=args.flow_hidden_dim,
            depth=args.flow_depth,
            n_heads=args.flow_heads,
            patch_size=args.flow_patch_size,
        ) for modality in MODALITIES
    }).to(device)
    register_gpt = None
    if args.stage == "generation":
        quantizer = tokenizer.alignment_model.register_modules[ModalityType.VISION].quantizer
        register_gpt = TextConditionedRegisterGPT(
            vocab_size=quantizer.vocab_size,
            text_dim=tokenizer.frozen_encoder.clip.transformer.width,
            hidden_dim=args.gpt_hidden_dim,
            max_registers=args.n_registers,
            depth=args.gpt_depth,
            n_heads=args.gpt_heads,
            dropout=args.gpt_dropout,
        ).to(device)
    system = FlexTokTrainingSystem(tokenizer, codec, flow_decoders, register_gpt).to(device)
    system.reconstruction_weight = args.reconstruction_weight
    system.flow_condition_dropout = args.flow_condition_dropout
    system.text_condition_dropout = args.text_condition_dropout

    if args.stage == "generation":
        if not args.alignment_checkpoint or not os.path.isfile(args.alignment_checkpoint):
            raise FileNotFoundError("--stage generation requires --alignment_checkpoint")
        checkpoint = torch.load(args.alignment_checkpoint, map_location="cpu")
        load_alignment_stack(system, checkpoint)
        for parameter in system.tokenizer.parameters():
            parameter.requires_grad = False
        for parameter in system.flow_decoders.parameters():
            parameter.requires_grad = False
        system.tokenizer.eval()
        system.flow_decoders.eval()
    return system


def _optimizer_parameters(module, weight_decay):
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias") or "embed" in name or "register_tokens" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _run_epoch(system, loader, optimizer, scaler, contrastive_loss, device, epoch, args, training):
    raw = system.module if hasattr(system, "module") else system
    if training:
        system.train()
        raw.codec.eval()
        raw.tokenizer.frozen_encoder.eval()
        if args.freeze_tokenizer:
            raw.tokenizer.eval()
        if args.stage == "generation":
            raw.tokenizer.eval()
            raw.flow_decoders.eval()
    else:
        system.eval()
    totals: Dict[str, float] = {}
    count = 0
    accum = max(args.accum_iter, 1)
    if training:
        optimizer.zero_grad(set_to_none=True)

    context = nullcontext if training else torch.no_grad
    with context():
        for step, raw_samples in enumerate(loader):
            samples = _move_samples(raw_samples, device)
            window_start = (step // accum) * accum
            window_size = min(accum, len(loader) - window_start)
            if training and step % accum == 0:
                lr_sched.adjust_learning_rate(optimizer, step / max(len(loader), 1) + epoch, args)
            amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with amp:
                if args.stage == "alignment":
                    loss, metrics, _ = system(samples, contrastive_loss, "alignment")
                else:
                    loss, metrics, _ = system(samples, None, "generation")
                scaled_loss = loss / window_size
            if training:
                scaler.scale(scaled_loss).backward()
                final_microbatch = step + 1 == len(loader)
                if (step + 1) % accum == 0 or final_microbatch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            batch = next(iter(samples.values())).shape[0]
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * batch
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach() if torch.is_tensor(value) else value) * batch
            count += batch
            if training and step % 20 == 0 and misc.is_main_process():
                print(f"Epoch {epoch} step {step}/{len(loader)} loss={float(loss):.5f}")
    return {key: _distributed_mean(value / max(count, 1), device) for key, value in totals.items()}


def _compact_state_dict(module):
    state = {}
    for key, value in module.state_dict().items():
        value = value.detach().cpu()
        state[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
    return state


def _checkpoint(system, args, epoch, metrics):
    raw = system.module if hasattr(system, "module") else system
    payload = {
        "format_version": 5 if args.stage == "generation" else 4,
        "stage": args.stage,
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "alignment_model": _compact_state_dict(raw.tokenizer.alignment_model),
    }
    if raw.flow_decoders is not None:
        payload["flow_decoders"] = _compact_state_dict(raw.flow_decoders)
    if raw.register_gpt is not None:
        payload["register_gpt"] = _compact_state_dict(raw.register_gpt)
    return payload


@torch.no_grad()
def _save_visualization(system, loader, device, epoch, args):
    if args.recon_vis_interval <= 0 or epoch % args.recon_vis_interval:
        return
    if args.stage not in {"alignment", "generation"}:
        return
    if args.stage == "generation":
        return _save_generation_visualization(system, loader, device, epoch, args)
    raw = system.module if hasattr(system, "module") else system
    raw.eval()
    samples = _move_samples(next(iter(loader)), device)
    latents = raw.encode_latents(samples, sample=False)
    output = raw.encode_registers(samples, nested_dropout=False, latents=latents)
    output_dir = Path(args.log_dir) / "reconstructions"
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_prefixes = [int(value) for value in str(args.recon_vis_prefixes).replace(",", " ").split()]
    prefix_lengths = sorted({k for k in requested_prefixes if 1 <= k <= args.n_registers})
    for modality in MODALITIES:
        target = _pixels_for_codec(samples[modality][:args.recon_vis_samples], modality)
        latent = latents[modality][:target.shape[0]]
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
        target_pixels = (target + 1) / 2
        vae_pixels = (raw.codec.decode(latent).clamp(-1, 1) + 1) / 2
        full_registers = output[f"{modality}_all_tokens_full"][:target.shape[0]]
        null_registers, null_mask = raw.flow_decoders[modality]._apply_null_condition(
            full_registers, None,
            torch.ones(target.shape[0], device=device, dtype=torch.bool),
        )
        unconditional = raw.flow_decoders[modality].sample(
            null_registers, latent.shape[1:], steps=args.flow_steps,
            noise=noise.clone(), guidance_scale=1.0, register_padding_mask=null_mask,
        )
        unconditional_pixels = (raw.codec.decode(unconditional).clamp(-1, 1) + 1) / 2
        rows = [("target", target_pixels), ("VAE", vae_pixels), ("uncond", unconditional_pixels)]
        prefix_metrics = []
        for k_keep in prefix_lengths:
            registers = output[f"{modality}_all_tokens_full"][:target.shape[0], :k_keep]
            sample = raw.flow_decoders[modality].sample(
                registers, latent.shape[1:], steps=args.flow_steps, noise=noise.clone(),
                guidance_scale=args.flow_guidance_scale,
            )
            decoded = (raw.codec.decode(sample).clamp(-1, 1) + 1) / 2
            rows.append((f"k={k_keep}", decoded))
            prefix_metrics.append({
                "k": k_keep,
                "latent_mse": float(torch.mean((sample - latent) ** 2)),
                **_image_metrics(decoded, target_pixels),
                "conditioning_delta": float(torch.mean((sample - unconditional) ** 2)),
            })
        figure, axes = plt.subplots(len(rows), target.shape[0], squeeze=False,
                                    figsize=(2.2 * target.shape[0], 2.2 * len(rows)))
        for row, (label, images) in enumerate(rows):
            for col, image in enumerate(images.float().cpu()):
                axes[row, col].imshow(image.permute(1, 2, 0).numpy())
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].text(
                        0.02, 0.98, label, transform=axes[row, col].transAxes,
                        va="top", ha="left", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                    )
        figure.tight_layout()
        figure.savefig(output_dir / f"epoch_{epoch:04d}_{modality}.png", dpi=140)
        plt.close(figure)
        for index, metric in enumerate(prefix_metrics):
            metric["monotonic_vs_previous"] = index == 0 or metric["latent_mse"] <= prefix_metrics[index - 1]["latent_mse"]
        shared_count = min(args.n_shared, full_registers.shape[1])
        ablations = {}
        for name, keep_slice in (
            ("shared_only", slice(0, shared_count)),
            ("private_only", slice(shared_count, full_registers.shape[1])),
        ):
            if keep_slice.start == keep_slice.stop:
                continue
            padding_mask = torch.ones(
                full_registers.shape[:2], device=device, dtype=torch.bool
            )
            padding_mask[:, keep_slice] = False
            ablated = raw.flow_decoders[modality].sample(
                full_registers, latent.shape[1:], steps=args.flow_steps,
                noise=noise.clone(), guidance_scale=args.flow_guidance_scale,
                register_padding_mask=padding_mask,
            )
            ablated_pixels = (raw.codec.decode(ablated).clamp(-1, 1) + 1) / 2
            ablations[name] = {
                "latent_mse": float(torch.mean((ablated - latent) ** 2)),
                **_image_metrics(ablated_pixels, target_pixels),
            }
        code_ids = output[f"{modality}_code_ids"][:target.shape[0]]
        diagnostics = {
            "flow_guidance_scale": args.flow_guidance_scale,
            "vae_roundtrip": _image_metrics(vae_pixels, target_pixels),
            "unconditioned": {
                "latent_mse": float(torch.mean((unconditional - latent) ** 2)),
                **_image_metrics(unconditional_pixels, target_pixels),
            },
            "shared_private_ablation": ablations,
            "unique_codes_per_register": [
                int(code_ids[:, index].unique().numel())
                for index in range(code_ids.shape[1])
            ],
            "prefixes": prefix_metrics,
        }
        with open(output_dir / f"epoch_{epoch:04d}_{modality}.json", "w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2)


def _image_metrics(prediction: torch.Tensor, target: torch.Tensor):
    mse_per_image = (prediction - target).square().flatten(1).mean(1)
    psnr = -10 * torch.log10(mse_per_image.clamp_min(1e-12))
    channels = prediction.shape[1]
    coords = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel = torch.exp(-(coords.square()) / (2 * 1.5 ** 2))
    kernel = (kernel / kernel.sum())[:, None] * (kernel / kernel.sum())[None, :]
    kernel = kernel.expand(channels, 1, 11, 11)
    mu_x = F.conv2d(prediction, kernel, padding=5, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=5, groups=channels)
    sigma_x = F.conv2d(prediction.square(), kernel, padding=5, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target.square(), kernel, padding=5, groups=channels) - mu_y.square()
    covariance = F.conv2d(prediction * target, kernel, padding=5, groups=channels) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2)
        * (sigma_x + sigma_y + 0.03 ** 2)
    )
    return {
        "pixel_mse": float(mse_per_image.mean()),
        "psnr": float(psnr.mean()),
        "ssim": float(ssim.flatten(1).mean(1).mean()),
    }


def _save_image_grid(path: Path, rows):
    count = rows[0][1].shape[0]
    figure, axes = plt.subplots(
        len(rows), count, squeeze=False, figsize=(2.2 * count, 2.2 * len(rows))
    )
    for row, (label, images) in enumerate(rows):
        for column, image in enumerate(images.float().cpu()):
            axes[row, column].imshow(image.permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].text(
                    0.02, 0.98, label, transform=axes[row, column].transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


@torch.no_grad()
def _save_generation_visualization(system, loader, device, epoch, args):
    raw = system.module if hasattr(system, "module") else system
    raw.eval()
    samples = _move_samples(next(iter(loader)), device)
    sample_count = min(args.recon_vis_samples, samples[ModalityType.VISION].shape[0])
    samples = {key: value[:sample_count] for key, value in samples.items()}
    latents = raw.encode_latents(samples, sample=False)
    register_output = raw.encode_registers(samples, nested_dropout=False, latents=latents)
    text_features, text_padding_mask = raw.encode_text(samples)
    requested = [int(value) for value in str(args.recon_vis_prefixes).replace(",", " ").split()]
    prefixes = sorted({value for value in requested if 1 <= value <= args.n_registers})
    if args.n_registers not in prefixes:
        prefixes.append(args.n_registers)
    reconstruction_dir = Path(args.log_dir) / "reconstructions"
    generation_dir = Path(args.log_dir) / "generations"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)

    for modality in MODALITIES:
        target = (_pixels_for_codec(samples[modality], modality) + 1) / 2
        vae = (raw.codec.decode(latents[modality]).clamp(-1, 1) + 1) / 2
        noise_generator = torch.Generator(device=device).manual_seed(args.seed + MODALITY_IDS[modality])
        noise = torch.randn(
            latents[modality].shape, device=device, dtype=latents[modality].dtype,
            generator=noise_generator,
        )
        gt_registers = register_output[f"{modality}_all_tokens_full"]
        exact_latent = raw.flow_decoders[modality].sample(
            gt_registers, latents[modality].shape[1:], steps=args.flow_steps,
            noise=noise.clone(), guidance_scale=args.flow_guidance_scale,
        )
        exact = (raw.codec.decode(exact_latent).clamp(-1, 1) + 1) / 2
        _save_image_grid(
            reconstruction_dir / f"epoch_{epoch:04d}_{modality}.png",
            [("target", target), ("VAE", vae), (f"exact k={args.n_registers}", exact)],
        )

        modality_ids = torch.full(
            (sample_count,), MODALITY_IDS[modality], device=device, dtype=torch.long
        )
        generated_codes = raw.register_gpt.generate(
            text_features, modality_ids, text_padding_mask,
            max_tokens=args.n_registers,
            temperature=args.generation_temperature,
            top_k=args.generation_top_k,
            guidance_scale=args.generation_guidance_scale,
            sample=True,
            generator=torch.Generator(device=device).manual_seed(args.seed + 100 + MODALITY_IDS[modality]),
        )
        quantizer = raw.tokenizer.alignment_model.register_modules[modality].quantizer
        generated_registers = quantizer.codes_to_quantized(generated_codes)
        generated_rows = [("target", target)]
        prefix_diagnostics = []
        for prefix in prefixes:
            generated_latent = raw.flow_decoders[modality].sample(
                generated_registers[:, :prefix], latents[modality].shape[1:],
                steps=args.flow_steps, noise=noise.clone(),
                guidance_scale=args.flow_guidance_scale,
            )
            generated_pixels = (raw.codec.decode(generated_latent).clamp(-1, 1) + 1) / 2
            generated_rows.append((f"text k={prefix}", generated_pixels))
            prefix_diagnostics.append({"k": prefix, **_image_metrics(generated_pixels, target)})
        _save_image_grid(
            generation_dir / f"epoch_{epoch:04d}_{modality}.png", generated_rows
        )
        diagnostics = {
            "kind": "text_to_discrete_registers_to_flow",
            "text_token_ids": samples[ModalityType.TEXT].squeeze(1).cpu().tolist(),
            "exact_reconstruction": _image_metrics(exact, target),
            "vae_roundtrip": _image_metrics(vae, target),
            "generated_prefixes": prefix_diagnostics,
        }
        with open(
            generation_dir / f"epoch_{epoch:04d}_{modality}.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(diagnostics, handle, indent=2)


def run_training(args, build_datasets: Callable, build_model: Callable, preprocessing_check: Callable):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    rank = misc.get_rank()
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    train_dataset, val_dataset = build_datasets(args)
    overfit_samples = 1 if args.overfit_one_sample else args.overfit_samples
    if overfit_samples > 0:
        # Use the same deterministic validation sample for both loaders. The
        # augmented training dataset changes on every access and cannot serve
        # as a strict reconstruction test.
        overfit_samples = min(overfit_samples, len(val_dataset))
        val_dataset = torch.utils.data.Subset(val_dataset, range(overfit_samples))
        train_dataset = val_dataset
    if not preprocessing_check(train_dataset):
        raise RuntimeError("Preprocessing sanity check failed")
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True) if args.distributed else torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=overfit_samples <= 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    system = _build_system(args, build_model, device)
    if args.distributed:
        system = torch.nn.parallel.DistributedDataParallel(system, device_ids=[args.gpu], find_unused_parameters=False)
    raw = system.module if hasattr(system, "module") else system
    if args.warm_start:
        warm = torch.load(args.warm_start, map_location="cpu")
        if warm.get("stage") != "alignment":
            raise ValueError("--warm_start requires an alignment checkpoint")
        saved_args = warm.get("args", {})
        for key in (
            "n_registers", "n_shared", "hidden_dim", "fsq_levels", "flow_depth",
            "tokenizer_input", "encoder_latent_patch_size",
        ):
            saved = saved_args.get(key, getattr(args, key))
            if str(saved) != str(getattr(args, key)):
                raise ValueError(f"Warm-start architecture mismatch for {key}")
        raw.tokenizer.alignment_model.load_state_dict(warm["alignment_model"], strict=True)
        raw.flow_decoders.load_state_dict(warm["flow_decoders"], strict=True)
        print(f"Warm-started tokenizer and flow decoders from {args.warm_start}")
    if args.freeze_tokenizer:
        if not args.warm_start:
            raise ValueError("--freeze_tokenizer requires --warm_start")
        for parameter in raw.tokenizer.parameters():
            parameter.requires_grad = False
        raw.tokenizer.eval()
        print("Frozen tokenizer; optimizing reconstruction flow decoders only")
    params = _optimizer_parameters(raw, args.weight_decay)
    trainable_parameters = sum(parameter.numel() for parameter in raw.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {trainable_parameters:,}")
    effective_batch = args.batch_size * args.accum_iter * misc.get_world_size()
    args.lr = args.lr or args.blr * effective_batch / 256
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    contrastive = CrossModalAlignmentLoss(
        MODALITIES,
        contrastive_weight=args.contrastive_weight,
        continuous_contrastive_weight=args.continuous_contrastive_weight,
        diversity_weight=args.diversity_weight,
        diversity_min_std=args.diversity_min_std,
    )

    best = {
        "flow": math.inf, "retrieval": math.inf, "joint": math.inf,
        "generation": math.inf,
    }
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        if resume.get("stage") != args.stage:
            raise ValueError(f"Resume stage {resume.get('stage')} does not match {args.stage}")
        saved_args = resume.get("args", {})
        architecture_keys = [
            "n_registers", "n_shared", "hidden_dim", "fsq_levels",
            "tokenizer_input", "encoder_latent_patch_size",
        ]
        architecture_keys += ["flow_depth"] if args.stage == "alignment" else [
            "gpt_hidden_dim", "gpt_depth", "gpt_heads",
        ]
        for key in architecture_keys:
            if str(saved_args.get(key)) != str(getattr(args, key)):
                raise ValueError(f"Resume architecture mismatch for {key}")
        raw.tokenizer.alignment_model.load_state_dict(resume["alignment_model"])
        if raw.flow_decoders is not None:
            raw.flow_decoders.load_state_dict(resume["flow_decoders"])
        if raw.register_gpt is not None:
            raw.register_gpt.load_state_dict(resume["register_gpt"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        best.update(resume.get("best", {}))
        args.start_epoch = int(resume["epoch"]) + 1
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(system, train_loader, optimizer, scaler, contrastive, device, epoch, args, True)
        val_metrics = _run_epoch(system, val_loader, optimizer, scaler, contrastive, device, epoch, args, False)
        if misc.is_main_process():
            _save_visualization(system, val_loader, device, epoch, args)
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            with open(Path(args.output_dir) / "log.txt", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if args.stage == "alignment":
                candidates = {
                    "flow": val_metrics["flow"],
                    "retrieval": -val_metrics.get("acc1", 0.0),
                    "joint": val_metrics["flow"] - 0.01 * val_metrics.get("acc1", 0.0),
                }
            else:
                candidates = {"generation": val_metrics["generation"]}
            improved_names = []
            for name, score in candidates.items():
                comparison = score
                current = best[name]
                improved = comparison < current
                if improved:
                    best[name] = comparison
                    improved_names.append(name)
            if improved_names:
                output_dir = Path(args.output_dir)
                temporary = output_dir / ".checkpoint_epoch_tmp.pth"
                torch.save(_checkpoint(system, args, epoch, val_metrics), temporary)
                for name in improved_names:
                    destination = output_dir / f"checkpoint_best_{name}.pth"
                    destination.unlink(missing_ok=True)
                    try:
                        os.link(temporary, destination)
                    except OSError:
                        shutil.copyfile(temporary, destination)
                temporary.unlink(missing_ok=True)
            if args.save_latest and (
                args.resume_interval <= 0 or (epoch + 1) % args.resume_interval == 0
            ):
                resume_payload = {
                    "stage": args.stage,
                    "epoch": epoch,
                    "args": vars(args),
                    "best": best,
                    "alignment_model": raw.tokenizer.alignment_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                if raw.flow_decoders is not None:
                    resume_payload["flow_decoders"] = raw.flow_decoders.state_dict()
                if raw.register_gpt is not None:
                    resume_payload["register_gpt"] = raw.register_gpt.state_dict()
                output_dir = Path(args.output_dir)
                temporary = output_dir / ".checkpoint_latest_tmp.pth"
                torch.save(resume_payload, temporary)
                os.replace(temporary, output_dir / "checkpoint_latest.pth")
            print(f"Epoch {epoch}: train={train_metrics} val={val_metrics}")
        if args.distributed:
            dist.barrier()
    if misc.is_main_process():
        print(f"Training completed in {(time.time() - start) / 3600:.2f} hours")
