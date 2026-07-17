# TVL-FlexTok Experiment Status

Last updated: 2026-07-17

This document records the current empirical state of TVL-FlexTok. Architecture
and usage belong in [README.md](README.md); design rationale and related work
belong in [MULTIMODAL_FUSION_MODEL_AUDIT.md](MULTIMODAL_FUSION_MODEL_AUDIT.md).

## Scope

The completed reconstruction experiment is an **eight-pair memorization
diagnostic**, not a held-out generalization result:

- Data: first eight records of the deterministic SSVTP validation split.
- Train and validation subsets contain the same eight static vision/tactile
  pairs. They are eight distinct image IDs, not one trajectory.
- Batch size: 8, producing one optimizer update per epoch.
- Tokenizer: 32 ordered FSQ registers, 8 shared registers, hidden size 512,
  four resampler layers, and FSQ levels `[8,8,8,5,5,5]`.
- Input: frozen FlexTok C4 VAE latent patches plus frozen TVL semantic patches.
- Reconstruction: separate depth-2 vision and tactile rectified-flow decoders
  operating in frozen VAE latent space.

The diagnostic establishes that the implementation can preserve noncollapsed
register codes and condition reconstruction. It does not establish performance
on unseen contacts or trajectories. The corrected full-data alignment run has
not yet produced a completed result.

## Training History

The tokenizer was learned during the initial joint phase. All continuations
froze it, set alignment/diversity weights to zero, and optimized only the flow
decoders with a fresh cosine learning-rate schedule.

| Phase | Slurm job | Updates | LR | Best validation flow loss |
| --- | ---: | ---: | ---: | ---: |
| Joint alignment | 9298734 | 300 | `3e-4` | 0.6459 |
| Flow continuation 1 | 9303223 | 1,000 | `1e-4` | 0.4587 |
| Flow continuation 2 | 9342982 | 1,000 | `5e-5` | 0.3521 |
| Flow continuation 3 | 9343764 | 1,000 | `2.5e-5` | **0.3075** |

Job `9343764` completed with exit code `0:0`. Its best joint/flow checkpoint is
epoch 994. The final phase decreased from validation flow loss 0.4000 at epoch
0 to 0.3075 at epoch 999; its last 20 epochs averaged 0.30755, so the restarted
schedule also reached a flat terminal window.

The four-phase curve is saved at:

```text
tvl_flextok/logs/runs/flextok_latent_overfit8_pos_flowft3/loss_curve_full_history.png
```

## Latest Reconstruction

The latest saved reconstruction grid is epoch 900 of phase 3. Metrics below use
all 32 registers and fixed visualization noise.

| Modality | Pixel MSE | PSNR | SSIM | Frozen-VAE PSNR ceiling |
| --- | ---: | ---: | ---: | ---: |
| Vision | 0.001730 | 27.62 dB | 0.9841 | 35.39 dB |
| Tactile | 0.000400 | 33.98 dB | 0.9947 | 40.13 dB |

Tactile prefix metrics improve monotonically from `k=1` through `k=32`.
Vision improves strongly at `k=16` and `k=32`, but `k=4` is slightly worse
than `k=1`; strict prefix monotonicity is therefore not yet achieved. The
vision grid still exhibits a latent-grid texture, despite accurate scene
layout and color recovery.

Persistent artifact root:

```text
tvl_flextok/logs/runs/flextok_latent_overfit8_pos_flowft3/
```

Important files:

```text
checkpoints/checkpoint_best_joint.pth
checkpoints/checkpoint_best_flow.pth
training_metrics.jsonl
reconstructions/epoch_0900_vision.png
reconstructions/epoch_0900_vision.json
reconstructions/epoch_0900_tactile.png
reconstructions/epoch_0900_tactile.json
loss_curve_full_history.png
```

Each compact checkpoint is about 121 MiB and excludes frozen TVL and VAE
weights. Scratch is only a training workspace; successful Slurm jobs archive
the best checkpoints, metrics, and reconstruction artifacts into the
persistent project path above.

## Interpretation

The third continuation confirms that optimizer restarts can continue fitting
the fixed eight-example decoder: full-prefix vision improved from 26.21 dB to
27.62 dB, and tactile from 32.78 dB to 33.98 dB. More repetitions of the same
depth-2 overfit schedule may still reduce memorization error, but they cannot
answer the important generalization question.

The next informative model experiment should use the intended depth-8 flow
decoder on a train/validation split with no duplicated examples. If vision
grid artifacts remain, compare latent patch size 2 against patch size 1 while
holding tokenizer capacity and evaluation noise fixed.

## Active Capacity Ablation

Slurm job `9348795` tests whether the 32-register bottleneck limits full-prefix
detail. It trains a new 64-register tokenizer from scratch on the same eight
pairs while holding shared registers (8), FSQ levels, hidden size, resampler
depth, latent patch size, and depth-2 flow decoder fixed. The 224x224 input
produces a 28x28 VAE latent; patch size 2 exposes 196 latent memory tokens, so
the change reduces compression from about 6.1x to about 3.1x.

The run uses 1,000 updates at learning rate `3e-4` and saves fixed-noise grids
for `k={1,4,8,16,32,64}`. At the time of this update it is running on
`babel-p9-32`. Scratch output will be archived to:

```text
tvl_flextok/logs/runs/flextok_latent_overfit8_reg64/
```
