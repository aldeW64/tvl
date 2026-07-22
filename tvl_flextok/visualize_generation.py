"""Render corrected text-to-register generation from a compact checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvl_flextok.engine import _build_system, _save_visualization, load_alignment_stack
from tvl_flextok.train import build_datasets, build_model, get_args_parser


def main():
    parser = argparse.ArgumentParser("Visualize TVL-FlexTok register generation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--datasets_dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["ssvtp"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--codec_cache_dir", default=None)
    parser.add_argument("--codec_local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--overfit_samples", type=int, default=8)
    parser.add_argument("--flow_steps", type=int, default=None)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    if checkpoint.get("stage") != "generation" or "register_gpt" not in checkpoint:
        raise ValueError("Expected a generation checkpoint containing register_gpt")

    args = get_args_parser().parse_args([])
    for key, value in checkpoint.get("args", {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.stage = "generation"
    args.stage1_checkpoint = cli.stage1_checkpoint
    args.datasets_dir = cli.datasets_dir
    args.datasets = cli.datasets
    args.device = cli.device
    args.codec_cache_dir = cli.codec_cache_dir
    args.codec_local_files_only = cli.codec_local_files_only
    args.recon_vis_samples = cli.n_samples
    args.recon_vis_interval = 1
    args.generation_top_k = 1
    args.log_dir = cli.output_dir
    if cli.flow_steps is not None:
        args.flow_steps = cli.flow_steps
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    _, validation = build_datasets(args)
    count = min(cli.overfit_samples, len(validation))
    validation = torch.utils.data.Subset(validation, range(count))
    loader = torch.utils.data.DataLoader(
        validation, batch_size=cli.n_samples, shuffle=False, num_workers=0
    )
    system = _build_system(args, build_model, device)
    load_alignment_stack(system, checkpoint)
    system.register_gpt.load_state_dict(checkpoint["register_gpt"], strict=True)
    system.eval()
    _save_visualization(system, loader, device, checkpoint.get("epoch", 0), args)
    print(f"Saved corrected generation panels under {args.log_dir}")


if __name__ == "__main__":
    main()
