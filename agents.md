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
| `tvl_flextok/train.py` | Main TVL-FlexTok train script for alignment and AR stages; supports both direct script and `python -m tvl_flextok.train` execution |
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

TVL-FlexTok reconstruction training:

```bash
python tvl_flextok/train.py \
  --stage ar \
  --feature_mode sequence \
  --tokenizer_input vae_tvl \
  --alignment_checkpoint /scratch/$USER/tvl_flextok/runs/flextok_alignment/checkpoint_best_joint.pth \
  --stage1_checkpoint /path/to/stage1.pth \
  --datasets_dir /path/to/data \
  --output_dir /scratch/$USER/tvl_flextok/runs/flextok_reconstruction
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
- TVL sequence mode returns unpooled patch tokens: OpenCLIP vision patch tokens before final pooling/projection and TIMM tactile ViT patch tokens from `forward_features()`. Text sequence mode is not implemented.
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
- `--stage ar` trains teacher-forced cross-modal next-register prediction. `--stage reconstruction` is only a deprecated alias.
- The latest completed reconstruction diagnostic is documented in `tvl_flextok/EXPERIMENT_STATUS.md`. As of 2026-07-17, `flextok_latent_overfit8_pos_flowft3` is the authoritative run: it memorizes eight static SSVTP validation pairs with a frozen tokenizer and depth-2 flow decoders. Do not present its metrics as held-out or trajectory-level generalization.
- The authoritative compact checkpoint is `tvl_flextok/logs/runs/flextok_latent_overfit8_pos_flowft3/checkpoints/checkpoint_best_joint.pth`; latest saved grids and JSON metrics are under the sibling `reconstructions/` directory.
- The phase-3 full-prefix epoch-900 diagnostic reaches 27.62 dB/0.9841 SSIM for vision and 33.98 dB/0.9947 SSIM for tactile. Vision is not strictly prefix-monotonic at `k=4`, and both modalities remain below the frozen VAE round-trip ceiling.
- Slurm job 9343764 completed successfully and archived phase-3 artifacts. No continuation should be described as pending.
- `tvl_flextok/configs/*.yaml` keys must exactly match argparse destination names. Unknown keys fail fast. Use names such as `blr`, `stage1_checkpoint`, `codec_id`, `flow_depth`, and `output_dir`.
- Avoid broad formatting-only changes in copied upstream code from MAE, OpenCLIP, ImageBind-LLM, or LLaMA Adapter unless the task is explicitly a cleanup.

## Validation Notes

- CPU-only quick validation is mainly `python tvl_flextok/test_modules.py`.
- Current TVL-FlexTok smoke coverage includes FSQ round trips/gradients, suffix-prefix invariance, paired nested dropout, pooled compatibility, latent-flow gradients, causal AR prediction, and strict config rejection.
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
- Alignment flow reconstruction uses the physically truncated active quantized prefix; do not pass a zero-padded suffix to decoder cross-attention.
- The verified frozen VAE cache defaults to `tvl_flextok/logs/models/flextok_vae_c4/`. Training checkpoints are staged under `/scratch/$USER/tvl_flextok/` and archived into `tvl_flextok/logs/runs/<name>/`; never leave the only checkpoint copy on scratch.

Known validation performed after the architecture update:

```bash
python tvl_flextok/test_modules.py
python -m py_compile tvl_enc/tvl.py tvl_flextok/train.py tvl_flextok/models/cross_modal_alignment.py \
  tvl_flextok/losses/alignment_loss.py tvl_flextok/losses/reconstruction_loss.py \
  tvl_flextok/test_modules.py tvl_flextok/visualize.py tvl_flextok/visualize_prefix_recon.py \
  tvl_flextok/validate_data.py tvl_flextok/sweep_registers.py tvl_flextok/visualizations/generate_all.py
python -m tvl_flextok.train --help
python tvl_flextok/train.py --help
```

Dataset-dependent integration tests were not run in this workspace because `./.datasets` was not available.
