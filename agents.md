# agents.md

This file is for coding agents working in this repository.

## Project Summary

This is the standalone Touch-Vision-Language (TVL) reference repository for *A Touch, Vision, and Language Dataset for Multimodal Alignment*. It contains:

- `tvl_enc/`: Stage 1 tactile encoder training. A TIMM tactile encoder is aligned with frozen OpenCLIP vision/text encoders using CLIP-style contrastive losses.
- `tvl_llama/`: Stage 3 TVL-LLaMA. A LLaMA-2-7B adapter consumes TVL vision/tactile features for multimodal language generation and benchmark evaluation.
- `tvl_flextok/`: TVL-FlexTok research package for ordered, FSQ-discrete multimodal register tokenization over frozen VAE latent patches and TVL semantic patches.
- `tvl_flextok/visualizations/`: Synthetic/report-driven TVL-FlexTok plots that can run without GPU, checkpoints, or dataset access.
- `assets/images/`: README and documentation images.

The parent MoT-Bagel repo treats this `tvl/` directory as a nested reference project. Keep changes here scoped unless the task explicitly asks to integrate with the parent project.

## Important Files

| Path | Purpose |
| --- | --- |
| `README.md` | Top-level setup, dataset, checkpoint, and citation information |
| `requirements.txt` | Python dependencies, including PyTorch CUDA 11.8 index and KNN_CUDA wheel |
| `setup.py` | Minimal editable install for `tvl_enc` and `tvl_llama` packages |
| `tvl_enc/main_pretrain.py` | Main Stage 1 tactile encoder training entry point |
| `tvl_enc/tvl.py` | `TVL` model: frozen OpenCLIP vision/text plus trainable tactile encoder |
| `tvl_enc/tacvis.py` | SSVTP/HCT dataset loading, transforms, tactile preprocessing, background subtraction |
| `tvl_enc/loss.py` | Pairwise multimodal CLIP-style losses and accuracy metrics |
| `tvl_enc/tools/visualize_affinity.py` | Stage 1 evaluation/affinity visualization |
| `tvl_llama/main_pretrain.py` | TVL-LLaMA pretraining entry point |
| `tvl_llama/main_finetune.py` | TVL-LLaMA finetuning entry point |
| `tvl_llama/evaluate.py` | GPT-based TVL benchmark evaluation |
| `tvl_llama/data/dataset.py` | Pretrain/finetune dataset readers for TVL, CC3M, Alpaca, and LLaVA-style data |
| `tvl_llama/llama/llama_adapter.py` | LLaMA adapter that projects TVL features into the LLaMA prefix path |
| `tvl_llama/exps/*.yaml` | Dataset config files with local paths that usually need editing |
| `tvl_flextok/README.md` | TVL-FlexTok overview |
| `tvl_flextok/EXPERIMENT_STATUS.md` | Latest completed runs, exact diagnostic scope, metrics, checkpoints, and reconstruction artifact roots |
| `tvl_flextok/MULTIMODAL_FUSION_MODEL_AUDIT.md` | Full model audit, multimodal fusion literature review, and LLM/VLA architecture roadmap |
| `tvl_flextok/docs/figures/tvl_flextok_full_process.svg` | Complete alignment-training, generation-training, and inference diagram |
| `tvl_flextok/train.py` | Main TVL-FlexTok train script for alignment and text-conditioned discrete-register generation; supports direct and module execution |
| `tvl_flextok/visualize_generation.py` | Post-training generation renderer that loads a compact generation checkpoint with current inference code |
| `tvl_flextok/models/` | Register token, cross-modal alignment, and decoder modules |
| `tvl_flextok/losses/` | Contrastive, flow-matching, and reconstruction diagnostic utilities |
| `tvl_flextok/configs/default.yaml` | Default TVL-FlexTok config |
| `tvl_flextok/test_modules.py` | CPU smoke tests for TVL-FlexTok modules |
| `tvl_flextok/visualizations/generate_all.py` | TVL-FlexTok synthetic/report visualization generator |

## Setup

Recommended environment from the README:

```bash
conda create -n tvl python=3.10 -y
conda activate tvl
conda install pytorch==2.1.2 cudatoolkit==11.8.0 -c pytorch -y
pip install packaging
pip install -r requirements.txt
pip install -e .
```

This project depends on large external assets that are not in the repo:

- TVL dataset from Hugging Face, usually with `ssvtp/` and `hct/` under one `--datasets_dir`.
- Optional revised dataset from `yoorhim/TVL-revise`.
- TVL encoder checkpoints from Hugging Face.
- LLaMA-2-7B weights for `tvl_llama/`, expected as `llama-2/llama-2-7b/*.pth` plus `llama-2/tokenizer.model`.
- CC3M, Alpaca GPT4, and LLaVA Instruct 150K metadata for TVL-LLaMA training.

Do not hardcode machine-specific dataset or checkpoint paths in library code. Prefer command-line arguments or the YAML files under `tvl_llama/exps/` and `tvl_flextok/configs/`.

## Common Commands

Stage 1 tactile encoder training:

```bash
cd tvl_enc
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 main_pretrain.py \
  --batch_size 256 --epochs 200 --warmup_epochs 10 --weight_decay 0.05 \
  --datasets ssvtp hct --active_modality_names vision tactile text \
  --find_unused_parameters --multi_epochs_dataloader \
  --log_name tvl_vittiny_tactile_encoder \
  --shuffle_text --no_text_prompt --replace_synonyms \
  --num_workers 20 --use_not_contact \
  --tactile_model vit_tiny_patch16_224 --blr 3e-4 \
  --datasets_dir /path/to/data
```

Stage 1 touch-vision evaluation:

```bash
cd tvl_enc
python -m tools.visualize_affinity \
  --checkpoint_path output_dir/tvl_vittiny_tactile_encoder/checkpoint-acc1.pth \
  --visualize_test --active_modality_names tactile vision \
  --tactile_model vit_tiny_patch16_224 --enable_flash_attention2 \
  --no_text_prompt --datasets ssvtp hct --seed 42 \
  --not_visualize --evaluate_all --datasets_dir /path/to/data
```

TVL-FlexTok smoke tests:

```bash
python tvl_flextok/test_modules.py
```

TVL-FlexTok training writes model artifacts under `/scratch/$USER/tvl_flextok/`.
On successful completion, Slurm scripts must move best checkpoints back to
`tvl_flextok/logs/runs/<name>/checkpoints/` alongside scalar logs and
reconstruction figures; do not leave the only copy on node-local scratch.

TVL-FlexTok alignment training:

```bash
python tvl_flextok/train.py \
  --stage alignment \
  --feature_mode sequence \
  --tokenizer_input vae_tvl \
  --stage1_checkpoint /path/to/stage1.pth \
  --datasets_dir /path/to/data \
  --output_dir /scratch/$USER/tvl_flextok/runs/flextok_alignment
```

TVL-FlexTok text-conditioned register generation:

```bash
python tvl_flextok/train.py \
  --stage generation \
  --alignment_checkpoint /scratch/$USER/tvl_flextok/runs/flextok_alignment/checkpoint_best_joint.pth \
  --stage1_checkpoint /path/to/stage1.pth \
  --datasets_dir /path/to/data \
  --output_dir /scratch/$USER/tvl_flextok/runs/flextok_generation
```

Corrected post-training generation visualization:

```bash
tvl_flextok/scripts/submit_slurm.sh generation_vis \
  GENERATION_CKPT=/path/to/checkpoint_best_generation.pth
```

TVL-LLaMA pretraining:

```bash
cd tvl_llama
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.launch \
  --master_port=1112 --nproc_per_node=4 --use_env main_pretrain.py \
  --data_config exps/pretrain-data-config.yaml \
  --batch_size 8 --seed 1 --epochs 150 --split_epoch 50 \
  --warmup_epochs 5 --blr 1.0e-4 --weight_decay 0.05 \
  --llama_path /path/to/llama-2 \
  --output_dir /path/to/output \
  --active_modality_names vision tactile \
  --checkpoint_path /path/to/tvl_encoder.pth \
  --tactile_model vit_base_patch16_224 \
  --crop_tacvis
```

Generate synthetic TVL-FlexTok visualizations:

```bash
python tvl_flextok/visualizations/generate_all.py
```

TVL-FlexTok module entrypoint help works in both modes:

```bash
python -m tvl_flextok.train --help
python tvl_flextok/train.py --help
```

## Development Guidance

- Run `git status --short --branch` before editing. This workspace may contain user changes.
- Use `rg`/`rg --files` for search.
- Most training and evaluation commands assume execution either from the relevant subdirectory (`tvl_enc` or `tvl_llama`) or from repo root with import paths adjusted by the script. TVL-FlexTok should use package imports (`tvl_flextok...`, `tvl_enc...`) and support both `python tvl_flextok/train.py` and `python -m tvl_flextok.train`.
- `tvl_enc.TVL.state_dict()` intentionally omits OpenCLIP weights and saves the tactile side plus scalar heads. Be careful when changing checkpoint loading/saving.
- `tvl_enc.TVL.forward()` is backward-compatible by default: without overrides it returns normalized pooled features. It now also accepts `feature_mode="pooled" | "sequence" | "both"`.
- TVL sequence mode returns unpooled contextual tokens: OpenCLIP vision/text tokens before final pooling and TIMM tactile ViT patch tokens from `forward_features()`.
- The OpenCLIP vision/text encoders are frozen by default in Stage 1; the tactile encoder is the primary trainable encoder.
- `TacVisDataset` is SSVTP; `TacVisDatasetV2` is HCT. The names are legacy and should not be casually renamed.
- Background subtraction has several code paths. Check `tvl_enc/tacvis.py` and pass `--subtract_background background` only when the dataset provides the expected background files.
- `TVLLoss` notes that the language-aware tactile-text loss is not designed for world size greater than 1.
- TVL-LLaMA adapter checkpoints intentionally exclude base LLaMA weights and embedded `image_bind.` weights. Keep this behavior unless changing checkpoint format deliberately.
- `tvl_llama/evaluate.py` uses the legacy `openai==0.28` package style and requires `OPENAI_API_KEY` for GPT-based judging.
- TVL-FlexTok code is research/experiment code. Read `tvl_flextok/README.md` and inspect the relevant implementation before changing register-token, feature-mode, prefix-reconstruction, or checkpoint logic.
- TVL-FlexTok defaults to `--feature_mode sequence`; use `--feature_mode pooled` only for ablations/debugging or backward-compatibility checks.
- `CrossModalAlignmentModel` and `RegisterTokenModule` expect `(B, N, D)` inputs internally. Pooled `(B, D)` features are only unsqueezed when the modality config explicitly says `feature_type: pooled`.
- TVL-FlexTok alignment always uses globally gathered shared-register contrastive loss plus vision/tactile rectified-flow reconstruction in frozen VAE latent space.
- The flow decoder is trained with register-conditioning dropout and sampled with classifier-free guidance. Shared-register contrastive alignment is applied after FSQ and through a lower-weight continuous pre-FSQ optimization bridge; neither objective reconstructs frozen TVL features.
- Feature reconstruction from register tokens back to frozen TVL encoder features is intentionally disabled. Register tokens are a compact bottleneck and should be free to discard frozen-encoder details that are not useful for alignment/reconstruction.
- Register modules use causal register self-attention followed by cross-attention to immutable TVL features. Never reintroduce joint input/register self-attention, which leaks suffix information into earlier prefixes.
- FSQ defaults to levels `[8,8,8,5,5,5]`; `*_code_ids` are the stable consumer interface and `*_all_tokens_full` contains quantized continuous embeddings.
- `--stage generation` freezes the tokenizer, flow decoders, VAE, and TVL/OpenCLIP encoders. One modality-conditioned causal Transformer predicts discrete FSQ register IDs from frozen contextual text tokens.
- Register GPT predicts exactly `R` IDs with no EOS class. Training inputs are `[modality BOS, target_id_1, ..., target_id_(R-1)]`; inference feeds each generated ID back before predicting the next. Preserve the empty-sequence-plus-placeholder implementation and its regression test when editing `generate()`.
- FSQ IDs must be converted back to quantized register embeddings before flow decoding. Registers condition the flow decoder; only the resulting VAE latent is passed to the frozen VAE decoder.
- Continuous latent-patch autoregression was removed. `--stage reconstruction` and `--stage ar` fail with a migration message; do not restore those objectives as FlexTok generation.
- Alignment nested dropout samples one prefix length per example uniformly over `1..R`, shares that length across each paired vision/tactile sample, and supplies a padding mask to flow cross-attention.
- Alignment flow conditioning dropout and classifier-free guidance use a learned null register condition, not all-zero register embeddings.
- The latest completed alignment reconstruction diagnostic is documented in `tvl_flextok/EXPERIMENT_STATUS.md`. `flextok_latent_overfit8_reg64_flowft3` memorizes eight static SSVTP validation pairs with a 64-register tokenizer and depth-2 flow decoders. Do not present its metrics as held-out or trajectory-level generalization.
- The authoritative alignment checkpoint is `tvl_flextok/logs/runs/flextok_latent_overfit8_reg64_flowft3/checkpoints/checkpoint_best_joint.pth`; its grids and JSON metrics are under the sibling `reconstructions/` directory.
- The 64-register phase-3 diagnostic reached best validation flow loss 0.2012. Its epoch-900 full-prefix results are 31.99 dB/0.9942 SSIM for vision and 37.60 dB/0.9977 for tactile; vision gains from `k=32` to `k=64` are small.
- `tvl_flextok/configs/*.yaml` keys must exactly match argparse destination names. Unknown keys fail fast. Use names such as `blr`, `stage1_checkpoint`, `codec_id`, `flow_depth`, and `output_dir`.
- Avoid broad formatting-only changes in copied upstream code from MAE, OpenCLIP, ImageBind-LLM, or LLaMA Adapter unless the task is explicitly a cleanup.

## Validation Notes

- CPU-only quick validation is mainly `python tvl_flextok/test_modules.py`.
- Current TVL-FlexTok smoke coverage includes FSQ round trips/gradients, suffix-prefix invariance, paired per-sample nested dropout, pooled compatibility, masked/null-conditioned latent flow, causal discrete-register GPT generation, checkpoint-metadata configuration, and strict config rejection.
- For import/entrypoint checks, run `python -m tvl_flextok.train --help` and `python tvl_flextok/train.py --help`.
- For config checks, load `tvl_flextok/configs/default.yaml` through `load_config_overrides()` and confirm it applies `feature_mode=sequence`, `blr`, `codec_id`, `flow_depth`, and `stage1_checkpoint`.
- Full Stage 1 and TVL-LLaMA validation require datasets, checkpoints, GPUs, and often LLaMA-2 access.
- If editing training code, prefer a lightweight import/smoke check first, then run the narrowest representative training/evaluation command available in the environment.
- If touching dataset path logic, test both `ssvtp` and `hct` paths when possible.

## Recent TVL-FlexTok Architecture Notes

The current TVL-FlexTok implementation is an FSQ-discrete register tokenizer over patch-token sequences:

- `TVL(..., feature_mode="pooled")` preserves legacy pooled normalized behavior.
- `TVL(..., feature_mode="sequence")` returns `(B, N, D)` patch-token features for vision and tactile.
- `TVL(..., feature_mode="both")` returns per-modality dictionaries with `pooled` and `sequence`.
- Vision sequence features use OpenCLIP ViT patch tokens before final pooling/projection.
- Tactile sequence features use TIMM ViT `forward_features()` with prefix/class tokens removed.
- TVL-FlexTok training builds modality configs from `args.feature_mode`; sequence dimensions are OpenCLIP visual width for vision and TIMM `num_features` for tactile.
- Alignment flow reconstruction uses full-shaped register tensors with explicit per-example padding masks; zero suffixes alone are not a valid attention mask.
- The verified frozen VAE cache defaults to `tvl_flextok/logs/models/flextok_vae_c4/`. Training checkpoints are staged under `/scratch/$USER/tvl_flextok/` and archived into `tvl_flextok/logs/runs/<name>/`; never leave the only checkpoint copy on scratch.
- Full-dataset alignment job `9413754` was cancelled because it preceded the per-sample dropout and masked-flow correction. Never resume or consume it; corrected alignment must start from scratch on one GPU.
- Corrected full-dataset one-GPU alignment job `9419434` is the accepted replacement. Its persistent artifact destination is `tvl_flextok/logs/runs/flextok_alignment_canonical_full_reg64_1gpu/` after successful completion.
- Eight-sample generation job `9420033` completed 2,000 epochs with exact token memorization. Its best epoch-1984 checkpoint is `tvl_flextok/logs/runs/flextok_canonical_generation_overfit8_reg64/checkpoints/checkpoint_best_generation.pth`.
- Never use job `9420033`'s original `generations/epoch_*.png` files as evidence: its resident inference loop duplicated the first ID and omitted the last. Corrected job `9420298` outputs are under the sibling `corrected_visualization/` directory and match the exact-register flow ceiling at `k=64`.

Known validation performed after the architecture update:

```bash
python tvl_flextok/test_modules.py
python -m py_compile tvl_enc/tvl.py tvl_flextok/train.py tvl_flextok/visualize_generation.py tvl_flextok/models/cross_modal_alignment.py \
  tvl_flextok/losses/alignment_loss.py tvl_flextok/losses/reconstruction_loss.py \
  tvl_flextok/test_modules.py tvl_flextok/visualize.py tvl_flextok/visualize_prefix_recon.py \
  tvl_flextok/validate_data.py tvl_flextok/sweep_registers.py tvl_flextok/visualizations/generate_all.py
python -m tvl_flextok.train --help
python tvl_flextok/train.py --help
```

Dataset-dependent GPU integration uses the shared SSVTP/HCT materialization at
`../datasets/tvl_dataset`; do not replace it with a duplicate download.
