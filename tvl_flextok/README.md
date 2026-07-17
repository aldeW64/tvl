# TVL-FlexTok

TVL-FlexTok compresses frozen TVL vision and tactile patch sequences into
ordered, discrete register tokens. The implementation follows the two-stage
FlexTok training pattern while retaining a cross-modal alignment objective.

## Architecture

```text
image -> frozen VAE -> latent patches ----+
                                          +-> causal register resampler -> FSQ codes
image -> frozen TVL -> semantic patches --+             |              |
                                                        |              +-> AR code model
                                                        +-> latent flow -> frozen VAE -> image
```

- Registers use causal self-attention and cross-attention to immutable encoder
  features. A suffix cannot influence an earlier register.
- Finite scalar quantization uses levels `[8,8,8,5,5,5]`, producing a 64,000
  token vocabulary without a learned codebook.
- Paired nested dropout retains the same prefix length for vision and tactile.
- The leading `n_shared` registers provide a globally gathered CLIP-style
  contrastive embedding. All retained registers condition reconstruction.
- A frozen `EPFL-VILAB/flextok_vae_c4` codec maps both modalities to continuous
  spatial latents. Its latent patches are encoder inputs, and separate modality
  flow decoders reconstruct those latents before frozen VAE decoding.
- Feature preservation and frozen-TVL feature reconstruction are intentionally
  absent; compression may discard irrelevant encoder information.

## Training Stages

`--stage alignment` always optimizes:

```text
contrastive_weight * global_cross_modal_contrastive
+ continuous_contrastive_weight * pre_FSQ_cross_modal_contrastive
+ diversity_weight * differentiable_fsq_anti_collapse
+ reconstruction_weight * mean(vision_flow, tactile_flow)
```

The diversity term applies only to pre-rounding FSQ scalars. It prevents all
samples/register positions from collapsing to one discrete code; it does not
reconstruct or preserve frozen TVL features.

The pre-FSQ contrastive term is an optimization bridge for the same shared
register objective, not feature preservation. Flow training drops register
conditioning on a configurable fraction of samples. Classifier-free guidance
is available through `--flow_guidance_scale`; it defaults to 1.0 because this
implementation does not yet reproduce FlexTok's APG norm-guidance algorithm.

`--stage ar` freezes the tokenizer and flow decoders. One teacher-forced causal
Transformer predicts tactile FSQ register codes from vision registers and
vision codes from tactile registers. Generated codes are converted back to
images by the corresponding flow decoder. `--stage reconstruction` remains a
deprecated alias for `ar`.

## Outputs

The tokenizer exposes, per modality:

- `*_continuous_tokens`: pre-quantization ordered registers.
- `*_all_tokens_full`: quantized full register sequence.
- `*_all_tokens`: the active nested-dropout prefix with a zero suffix.
- `*_code_ids`: mixed-radix FSQ IDs in `[0, 64000)`.
- `*_shared`: normalized retrieval embedding.

Alignment saves `checkpoint_best_flow.pth`,
`checkpoint_best_retrieval.pth`, and `checkpoint_best_joint.pth`. Checkpoints
contain tokenizer/flow weights and architecture metadata, but omit frozen TVL
and VAE weights.

## Commands

```bash
python tvl_flextok/test_modules.py

tvl_flextok/scripts/submit_slurm.sh alignment
tvl_flextok/scripts/submit_slurm.sh alignment_4gpu
tvl_flextok/scripts/submit_slurm.sh alignment \
  N_REGISTERS=64 N_SHARED=8 RECON_VIS_PREFIXES="1 4 8 16 32 64"
tvl_flextok/scripts/submit_slurm.sh reconstruction \
  ALIGNMENT_CKPT=/path/to/checkpoint_best_joint.pth
```

`N_REGISTERS` and `N_SHARED` configure register capacity in both alignment
Slurm launchers. Increasing the register count requires training a new
tokenizer; a 32-register checkpoint cannot be strictly warm-started into a
64-register model because its learned register and position tensors differ.

Direct execution and module execution are both supported:

```bash
python tvl_flextok/train.py --help
python -m tvl_flextok.train --help
```

Slurm stdout/stderr use `tvl_flextok/logs/slurm/`; each experiment writes
reconstruction grids and application logs below `tvl_flextok/logs/runs/`.
Training stages checkpoints under `/scratch/$USER/tvl_flextok/`, while the
verified VAE codec cache defaults to `tvl_flextok/logs/models/`. After
successful training, supplied Slurm scripts move best checkpoints into the
run's `tvl_flextok/logs/runs/<name>/checkpoints/` folder alongside scalar logs
and reconstruction artifacts. W&B defaults to offline mode.

Use `--tokenizer_input vae_tvl` (the default) for latent patches plus TVL
semantic patches, `vae` for the closest original-FlexTok encoder path, or `tvl`
only as an ablation. Reconstruction grids contain target, direct VAE
round-trip, zero-register baseline, and every configured register prefix. JSON
diagnostics report VAE fidelity and whether the flow decoder uses register
conditioning.

See [MULTIMODAL_FUSION_MODEL_AUDIT.md](MULTIMODAL_FUSION_MODEL_AUDIT.md) for
the model audit and guidance for future LLM/VLA consumers. See
[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) for the latest completed runs,
loss curves, checkpoints, reconstruction metrics, and their interpretation
limits.
