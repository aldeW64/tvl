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

from tvl_enc.tacvis import RGB_MEAN, RGB_STD, TAC_MEAN, TAC_STD
from tvl_enc.tvl import ModalityType
from tvl_enc.util import lr_sched, misc
from tvl_flextok.losses.alignment_loss import CrossModalAlignmentLoss
from tvl_flextok.models.autoregressive_decoder import AutoregressiveDecoder
from tvl_flextok.models.flow_decoder import FrozenVAECodec, LatentFlowDecoder


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
    def __init__(self, tokenizer, codec, flow_decoders, ar_model=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.codec = codec
        self.flow_decoders = flow_decoders
        self.ar_model = ar_model

    def encode_latents(self, samples, sample=False):
        return {
            modality: self.codec.encode(_pixels_for_codec(samples[modality], modality), sample=sample)
            for modality in MODALITIES
        }

    def encode_registers(self, samples, nested_dropout=None, latents=None):
        with torch.no_grad():
            frozen = self.tokenizer.encode(samples)
            if latents is None and self.tokenizer.alignment_model.tokenizer_input in {"vae", "vae_tvl"}:
                latents = self.encode_latents(samples, sample=False)
        return self.tokenizer.alignment_model(
            frozen, apply_nested_dropout=nested_dropout, latent_dict=latents
        )

    def forward(self, samples, contrastive_loss=None, stage="alignment"):
        if stage == "alignment":
            return self.alignment_losses(samples, contrastive_loss)
        return self.ar_losses(samples)

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
            active_registers = output[f"{modality}_all_tokens"][:, :output["k_keep"]]
            generator = None
            if not self.training:
                generator = torch.Generator(device=targets[modality].device).manual_seed(
                    1701 + MODALITY_IDS[modality]
                )
            flow = self.flow_decoders[modality].flow_loss(
                targets[modality], active_registers, generator=generator,
                condition_dropout=self.flow_condition_dropout if self.training else 0.0,
            )
            losses[f"flow_{modality}"] = flow
            total_flow = total_flow + flow
        losses["flow"] = total_flow / len(MODALITIES)
        for modality in MODALITIES:
            codes = output[f"{modality}_code_ids"]
            losses[f"code_utilization_{modality}"] = codes.unique().numel() / max(codes.numel(), 1)
        # Keep every ordered-register parameter in the DDP graph even when a
        # sampled short prefix correctly gives the suffix zero gradient.
        graph_anchor = sum(output[f"{modality}_continuous_tokens"].sum() for modality in MODALITIES) * 0
        return align["total_loss"] + self.reconstruction_weight * losses["flow"] + graph_anchor, losses, output

    def ar_losses(self, samples):
        with torch.no_grad():
            output = self.encode_registers(samples, nested_dropout=False)
        max_registers = output["vision_code_ids"].shape[1]
        prefix_values = self.tokenizer.alignment_model.register_modules["vision"]._k_keep_values
        k_keep = prefix_values[torch.randint(len(prefix_values), ()).item()] if self.training else max_registers
        losses = {}
        for source, target in ((ModalityType.VISION, ModalityType.TACTILE),
                               (ModalityType.TACTILE, ModalityType.VISION)):
            target_ids = output[f"{target}_code_ids"][:, :k_keep]
            target_modality = torch.full(
                (target_ids.shape[0],), MODALITY_IDS[target], device=target_ids.device, dtype=torch.long
            )
            losses[f"ar_{source}_to_{target}"] = self.ar_model.loss(
                output[f"{source}_all_tokens_full"], target_ids, target_modality
            )
        losses["ar"] = sum(losses.values()) / 2
        return losses["ar"], losses, output


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
    quantizer = tokenizer.alignment_model.register_modules[ModalityType.VISION].quantizer
    ar_model = AutoregressiveDecoder(
        vocab_size=quantizer.vocab_size,
        register_dim=args.hidden_dim,
        hidden_dim=args.ar_hidden_dim,
        max_registers=args.n_registers,
        depth=args.ar_depth,
        n_heads=args.ar_heads,
    ).to(device) if args.stage == "ar" else None
    system = FlexTokTrainingSystem(tokenizer, codec, flow_decoders, ar_model).to(device)
    system.reconstruction_weight = args.reconstruction_weight
    system.flow_condition_dropout = args.flow_condition_dropout

    if args.stage == "ar":
        if not args.alignment_checkpoint or not os.path.isfile(args.alignment_checkpoint):
            raise FileNotFoundError("--stage ar requires --alignment_checkpoint")
        checkpoint = torch.load(args.alignment_checkpoint, map_location="cpu")
        system.tokenizer.alignment_model.load_state_dict(checkpoint["alignment_model"], strict=True)
        system.flow_decoders.load_state_dict(checkpoint["flow_decoders"], strict=True)
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
        if args.stage == "ar":
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
                    loss, metrics, _ = system(samples, None, "ar")
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
        "format_version": 3,
        "stage": args.stage,
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "alignment_model": _compact_state_dict(raw.tokenizer.alignment_model),
        "flow_decoders": _compact_state_dict(raw.flow_decoders),
    }
    if raw.ar_model is not None:
        payload["ar_model"] = _compact_state_dict(raw.ar_model)
    return payload


@torch.no_grad()
def _save_visualization(system, loader, device, epoch, args):
    if args.stage != "alignment" or args.recon_vis_interval <= 0 or epoch % args.recon_vis_interval:
        return
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
        vae_mse = torch.mean((vae_pixels - target_pixels) ** 2)
        full_registers = output[f"{modality}_all_tokens_full"][:target.shape[0]]
        unconditional = raw.flow_decoders[modality].sample(
            torch.zeros_like(full_registers), latent.shape[1:], steps=args.flow_steps,
            noise=noise.clone(), guidance_scale=1.0,
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
            mse = torch.mean((decoded - target_pixels) ** 2)
            mu_x, mu_y = decoded.mean(), target_pixels.mean()
            var_x, var_y = decoded.var(correction=0), target_pixels.var(correction=0)
            covariance = ((decoded - mu_x) * (target_pixels - mu_y)).mean()
            ssim = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
                (mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2)
            )
            prefix_metrics.append({
                "k": k_keep,
                "latent_mse": float(torch.mean((sample - latent) ** 2)),
                "pixel_mse": float(mse),
                "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
                "ssim": float(ssim),
                "conditioning_delta": float(torch.mean((sample - unconditional) ** 2)),
            })
        figure, axes = plt.subplots(len(rows), target.shape[0], squeeze=False,
                                    figsize=(2.2 * target.shape[0], 2.2 * len(rows)))
        for row, (label, images) in enumerate(rows):
            for col, image in enumerate(images.float().cpu()):
                axes[row, col].imshow(image.permute(1, 2, 0).numpy())
                axes[row, col].axis("off")
                if col == 0:
                    axes[row, col].set_ylabel(label)
        figure.tight_layout()
        figure.savefig(output_dir / f"epoch_{epoch:04d}_{modality}.png", dpi=140)
        plt.close(figure)
        for index, metric in enumerate(prefix_metrics):
            metric["monotonic_vs_previous"] = index == 0 or metric["latent_mse"] <= prefix_metrics[index - 1]["latent_mse"]
        diagnostics = {
            "flow_guidance_scale": args.flow_guidance_scale,
            "vae_roundtrip": {
                "pixel_mse": float(vae_mse),
                "psnr": float(-10 * torch.log10(vae_mse.clamp_min(1e-12))),
            },
            "unconditioned": {
                "latent_mse": float(torch.mean((unconditional - latent) ** 2)),
                "pixel_mse": float(torch.mean((unconditional_pixels - target_pixels) ** 2)),
            },
            "prefixes": prefix_metrics,
        }
        with open(output_dir / f"epoch_{epoch:04d}_{modality}.json", "w", encoding="utf-8") as handle:
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

    best = {"flow": math.inf, "retrieval": math.inf, "joint": math.inf, "ar": math.inf}
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        if resume.get("stage") != args.stage:
            raise ValueError(f"Resume stage {resume.get('stage')} does not match {args.stage}")
        saved_args = resume.get("args", {})
        for key in (
            "n_registers", "n_shared", "hidden_dim", "fsq_levels", "flow_depth",
            "tokenizer_input", "encoder_latent_patch_size",
        ):
            if str(saved_args.get(key)) != str(getattr(args, key)):
                raise ValueError(f"Resume architecture mismatch for {key}")
        raw.tokenizer.alignment_model.load_state_dict(resume["alignment_model"])
        raw.flow_decoders.load_state_dict(resume["flow_decoders"])
        if raw.ar_model is not None:
            raw.ar_model.load_state_dict(resume["ar_model"])
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
                candidates = {"ar": val_metrics["ar"]}
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
            if args.resume_interval > 0 and (epoch + 1) % args.resume_interval == 0:
                scratch = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "tvl_flextok" / os.environ.get("SLURM_JOB_ID", "local")
                scratch.mkdir(parents=True, exist_ok=True)
                resume_payload = {
                    "stage": args.stage,
                    "epoch": epoch,
                    "args": vars(args),
                    "best": best,
                    "alignment_model": raw.tokenizer.alignment_model.state_dict(),
                    "flow_decoders": raw.flow_decoders.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                if raw.ar_model is not None:
                    resume_payload["ar_model"] = raw.ar_model.state_dict()
                torch.save(resume_payload, scratch / "checkpoint_resume.pth")
            print(f"Epoch {epoch}: train={train_metrics} val={val_metrics}")
        if args.distributed:
            dist.barrier()
    if misc.is_main_process():
        print(f"Training completed in {(time.time() - start) / 3600:.2f} hours")
