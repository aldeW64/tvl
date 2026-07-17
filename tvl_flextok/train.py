"""TVL-FlexTok training entry point."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvl_enc.tacvis import (
    RGB_AUGMENTS,
    RGB_MEAN,
    RGB_PREPROCESS,
    RGB_STD,
    TAC_AUGMENTS,
    TAC_AUGMENTS_BG,
    TAC_MEAN,
    TAC_PREPROCESS,
    TAC_STD,
    TacVisDataset,
    TacVisDatasetV2,
)
from tvl_enc.tvl import ModalityType, TVL
from tvl_flextok.models.cross_modal_alignment import CrossModalAlignmentModel, FlexTokWrapper


DEFAULT_SCRATCH_ROOT = Path("/scratch") / os.environ.get("USER", "peilinwu") / "tvl_flextok"


def get_args_parser():
    parser = argparse.ArgumentParser("TVL-FlexTok")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_registers", type=int, default=32)
    parser.add_argument("--n_shared", "--n_alignment_registers", dest="n_shared", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--model_dropout", type=float, default=0.1)
    parser.add_argument("--feature_mode", choices=["pooled", "sequence"], default="sequence")
    parser.add_argument("--tokenizer_input", choices=["vae", "tvl", "vae_tvl"], default="vae_tvl")
    parser.add_argument("--nested_dropout", action="store_true", default=True)
    parser.add_argument("--no_nested_dropout", action="store_false", dest="nested_dropout")
    parser.add_argument("--nested_dropout_mode", choices=["power_of_two", "uniform"], default="power_of_two")
    parser.add_argument("--fsq_levels", default="8 8 8 5 5 5")
    parser.add_argument("--use_token_type_embed", action="store_true", default=True)
    parser.add_argument("--no_token_type_embed", action="store_false", dest="use_token_type_embed")

    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--continuous_contrastive_weight", type=float, default=0.25)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--diversity_min_std", type=float, default=0.2)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--codec_id", default="EPFL-VILAB/flextok_vae_c4")
    parser.add_argument("--codec_cache_dir", default=None)
    parser.add_argument("--codec_local_files_only", action="store_true")
    parser.add_argument("--encoder_latent_patch_size", type=int, default=2)
    parser.add_argument("--flow_hidden_dim", type=int, default=512)
    parser.add_argument("--flow_depth", type=int, default=8)
    parser.add_argument("--flow_heads", type=int, default=8)
    parser.add_argument("--flow_patch_size", type=int, default=2)
    parser.add_argument("--flow_steps", type=int, default=25)
    parser.add_argument("--flow_condition_dropout", type=float, default=0.1)
    parser.add_argument("--flow_guidance_scale", type=float, default=1.0)
    parser.add_argument("--ar_hidden_dim", type=int, default=512)
    parser.add_argument("--ar_depth", type=int, default=8)
    parser.add_argument("--ar_heads", type=int, default=8)

    parser.add_argument("--stage", choices=["alignment", "ar", "reconstruction"], default="alignment")
    parser.add_argument("--alignment_checkpoint", default=None)
    parser.add_argument("--stage1_checkpoint", required=False, default=None)
    parser.add_argument("--tactile_model", default="vit_tiny_patch16_224",
                        choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224"])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--datasets_dir", default="./.datasets")
    parser.add_argument("--datasets", nargs="+", default=["ssvtp", "hct"], choices=["ssvtp", "hct"])
    parser.add_argument("--subtract_background", default=None, choices=[None, "mean", "median", "background"])
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--overfit_one_sample", action="store_true")
    parser.add_argument("--overfit_samples", type=int, default=0,
                        help="Use the same N deterministic validation samples for train and validation")

    parser.add_argument("--output_dir", default=str(DEFAULT_SCRATCH_ROOT / "runs" / "default"))
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--log_name", default=None)
    parser.add_argument("--recon_vis_interval", type=int, default=5)
    parser.add_argument("--recon_vis_samples", type=int, default=4)
    parser.add_argument("--recon_vis_prefixes", default="1 4 8 16 32")
    parser.add_argument("--save_latest", action="store_true", default=False)
    parser.add_argument("--no_save_latest", action="store_false", dest="save_latest")
    parser.add_argument("--resume", default="")
    parser.add_argument("--warm_start", default="",
                        help="Load compact tokenizer/flow weights but start a fresh optimizer schedule")
    parser.add_argument("--freeze_tokenizer", action="store_true",
                        help="Freeze register tokenizer during alignment reconstruction fine-tuning")
    parser.add_argument("--resume_interval", type=int, default=10,
                        help="Write node-local optimizer state every N epochs; 0 disables it")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="tvl-flextok")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--config", default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_eval", action="store_true")
    return parser


def build_datasets(args):
    modalities = [ModalityType.VISION, ModalityType.TACTILE]
    prompt = "This image gives tactile feelings of "
    train, val = [], []
    if "ssvtp" in args.datasets:
        tactile_train = TAC_AUGMENTS_BG if args.subtract_background == "background" else TAC_AUGMENTS
        root = os.path.join(args.datasets_dir, "ssvtp")
        train.append(TacVisDataset(
            root_dir=root, split="train", transform_rgb=RGB_AUGMENTS,
            transform_tac=tactile_train, modality_types=modalities, text_prompt=prompt,
        ))
        val.append(TacVisDataset(
            root_dir=root, split="val", transform_rgb=RGB_PREPROCESS,
            transform_tac=TAC_PREPROCESS, modality_types=modalities, text_prompt=prompt,
        ))
    if "hct" in args.datasets:
        root = os.path.join(args.datasets_dir, "hct")
        directories = sorted(
            os.path.join(root, name) for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name)) and os.path.isfile(os.path.join(root, name, "contact.json"))
        ) if os.path.isdir(root) else []
        if directories:
            tactile_train = TAC_AUGMENTS if args.subtract_background is None else None
            train.append(TacVisDatasetV2(
                root_dir=directories, split="train", transform_rgb=RGB_AUGMENTS,
                transform_tac=tactile_train, modality_types=modalities, text_prompt=prompt,
            ))
            val.append(TacVisDatasetV2(
                root_dir=directories, split="val", transform_rgb=RGB_PREPROCESS,
                transform_tac=TAC_PREPROCESS, modality_types=modalities, text_prompt=prompt,
            ))
    if not train:
        raise ValueError(f"No requested datasets found below {args.datasets_dir}")
    return (train[0], val[0]) if len(train) == 1 else (ConcatDataset(train), ConcatDataset(val))


def _sanity_check_preprocessing(dataset):
    sample = dataset[0]
    for modality, mean, std in (
        (ModalityType.VISION, RGB_MEAN, RGB_STD),
        (ModalityType.TACTILE, TAC_MEAN, TAC_STD),
    ):
        tensor = sample[modality]
        if isinstance(tensor, list):
            tensor = tensor[0]
        tensor = tensor.squeeze()
        if tensor.shape != (3, 224, 224) or not torch.isfinite(tensor).all():
            return False
        mean_tensor = tensor.new_tensor(mean).view(3, 1, 1)
        std_tensor = tensor.new_tensor(std).view(3, 1, 1)
        pixels = tensor * std_tensor + mean_tensor
        if pixels.max() < 0.05 or pixels.min() > 0.95:
            return False
    return True


def _load_stage1(model, checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("A valid --stage1_checkpoint is required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    tactile_keys = [key for key in state if "tactile" in key]
    if not tactile_keys:
        raise ValueError("Stage-1 checkpoint contains no tactile encoder weights")
    result = model.load_state_dict(state, strict=False)
    expected_tactile = [name for name, _ in model.tactile_encoder.named_parameters()]
    missing_tactile = [name for name in expected_tactile if any(
        missing.endswith(f"tactile_encoder.{name}") for missing in result.missing_keys
    )]
    if len(missing_tactile) == len(expected_tactile):
        raise ValueError("Stage-1 checkpoint did not load any tactile encoder parameters")
    print(f"Loaded Stage-1 checkpoint {checkpoint_path}; missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")


def build_model(args, device):
    frozen = TVL(
        tactile_model=args.tactile_model,
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
        feature_mode=args.feature_mode,
    )
    _load_stage1(frozen, args.stage1_checkpoint)
    if args.feature_mode == "sequence":
        vision_dim = frozen.clip.visual.conv1.out_channels
        tactile_dim = frozen.tactile_encoder.num_features
    else:
        vision_dim = getattr(frozen.clip.visual, "output_dim", 768)
        tactile_dim = frozen.tactile_encoder.num_features
    configs = {
        ModalityType.VISION: {
            "input_dim": vision_dim, "feature_type": args.feature_mode,
            "latent_channels": 4, "latent_patch_size": args.encoder_latent_patch_size,
        },
        ModalityType.TACTILE: {
            "input_dim": tactile_dim, "feature_type": args.feature_mode,
            "latent_channels": 4, "latent_patch_size": args.encoder_latent_patch_size,
        },
    }
    levels = tuple(int(value) for value in str(args.fsq_levels).replace(",", " ").split())
    alignment = CrossModalAlignmentModel(
        configs, args.hidden_dim, args.n_registers, args.n_shared, args.n_layers,
        args.n_heads, args.model_dropout, args.nested_dropout,
        args.nested_dropout_mode, use_token_type_embed=args.use_token_type_embed,
        fsq_levels=levels, tokenizer_input=args.tokenizer_input,
    )
    return FlexTokWrapper(frozen, alignment, args.feature_mode).to(device)


def load_config_overrides(args, config_path, argv):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parser = get_args_parser()
    valid = {action.dest for action in parser._actions}
    explicit = {action.dest for action in parser._actions if any(option in argv for option in action.option_strings)}
    unknown, values = [], {}
    for section, content in config.items():
        items = content.items() if isinstance(content, dict) else [(section, content)]
        for key, value in items:
            if key not in valid:
                unknown.append(f"{section}.{key}" if isinstance(content, dict) else key)
            else:
                values[key] = value
    if unknown:
        raise KeyError("Unknown config key(s): " + ", ".join(unknown))
    for key, value in values.items():
        if key not in explicit:
            setattr(args, key, value)
    return args


def main():
    args = get_args_parser().parse_args()
    if args.config:
        args = load_config_overrides(args, args.config, sys.argv)
    if args.stage == "reconstruction":
        print("WARNING: --stage reconstruction is deprecated; using --stage ar")
        args.stage = "ar"
    if args.stage == "alignment" and args.reconstruction_weight <= 0:
        raise ValueError("Alignment requires --reconstruction_weight > 0")
    if not 0 <= args.flow_condition_dropout < 1:
        raise ValueError("--flow_condition_dropout must be in [0, 1)")
    if args.flow_guidance_scale < 1:
        raise ValueError("--flow_guidance_scale must be >= 1")
    if args.continuous_contrastive_weight < 0 or args.diversity_weight < 0:
        raise ValueError("Tokenizer auxiliary loss weights must be non-negative")
    if args.log_name:
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    args.log_dir = args.log_dir or args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    from tvl_flextok.engine import run_training
    run_training(args, build_datasets, build_model, _sanity_check_preprocessing)


if __name__ == "__main__":
    main()
