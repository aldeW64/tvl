"""Render fixed-noise flow reconstructions from an alignment checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvl_flextok.engine import _build_system, _save_visualization
from tvl_flextok.train import build_datasets, build_model, get_args_parser


def main():
    parser = argparse.ArgumentParser("Visualize TVL-FlexTok flow reconstruction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--datasets_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["ssvtp"])
    parser.add_argument("--output_dir", default="tvl_flextok/logs/manual_visualization")
    parser.add_argument("--codec_cache_dir", default=None)
    parser.add_argument("--codec_local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--flow_guidance_scale", type=float, default=None)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    if checkpoint.get("stage") != "alignment" or "flow_decoders" not in checkpoint:
        raise ValueError("Expected a format-v2 alignment checkpoint with flow decoders")
    args = get_args_parser().parse_args([])
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint.get("format_version", 1) < 3 and "tokenizer_input" not in checkpoint_args:
        args.tokenizer_input = "tvl"
    for key, value in checkpoint_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.stage = "alignment"
    args.stage1_checkpoint = cli.stage1_checkpoint
    args.datasets_dir = cli.datasets_dir
    args.datasets = cli.datasets
    args.device = cli.device
    args.codec_cache_dir = cli.codec_cache_dir
    args.codec_local_files_only = cli.codec_local_files_only
    args.recon_vis_samples = cli.n_samples
    args.recon_vis_interval = 1
    args.flow_steps = cli.flow_steps or args.flow_steps
    if cli.flow_guidance_scale is not None:
        args.flow_guidance_scale = cli.flow_guidance_scale
    args.log_dir = cli.output_dir
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _, validation = build_datasets(args)
    loader = torch.utils.data.DataLoader(validation, batch_size=cli.n_samples, shuffle=False, num_workers=0)
    system = _build_system(args, build_model, device)
    system.tokenizer.alignment_model.load_state_dict(checkpoint["alignment_model"], strict=True)
    system.flow_decoders.load_state_dict(checkpoint["flow_decoders"], strict=True)
    system.eval()
    _save_visualization(system, loader, device, checkpoint.get("epoch", 0), args)
    print(f"Saved reconstructions under {Path(args.log_dir) / 'reconstructions'}")


if __name__ == "__main__":
    main()
