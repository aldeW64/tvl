# TVL-FlexTok Experiment Status

Last updated: 2026-07-22

## Canonical FlexTok Correction

The earlier continuous latent-patch autoregressive experiments (`9402699`,
`9402722`, and `9402723`) are retained only as historical failed ablations.
Teacher-forced latent NLL was not a valid measure of free-running visual
quality, and that decoder was not FlexTok's generative second stage. The code
path has been removed.

The corrected implementation has two stages. Alignment learns ordered FSQ
registers and rectified-flow detokenizers with per-sample uniform nested
dropout, paired prefix lengths, masked cross-attention, and a learned null
condition. Generation freezes that stack and trains one text-conditioned
causal Transformer to predict discrete FSQ IDs for either vision or tactile.
Exact source reconstruction and free-running text generation are saved in
separate folders.

Full-data job `9413754` was cancelled after 1:26:21 because it used the
superseded batch-wide prefix and zero-suffix flow conditioning. It must not be
resumed or used as a corrected alignment checkpoint. After GPU smoke
verification, it was replaced from scratch by active corrected job `9419434`.

Corrected alignment smoke job `9419399` completed on one GPU with eight SSVTP
examples. It exercised one train update, deterministic validation, per-example
prefix masks, learned-null flow conditioning, best-checkpoint selection, and
persistent artifact archival. Generation smoke `9419432` then
exercised deterministic caption loading, contextual OpenCLIP text tokens,
discrete-register cross entropy, and format-version-5 checkpoint writing. Its
metrics exposed overlarge tied-embedding initialization; that initialization
was corrected. Final generation smoke `9419453` completed with validation
cross entropy 11.037 and perplexity 62,138, matching the expected untrained
64,000-way baseline and confirming the repaired initialization and checkpoint
path.

Generation visualization smoke `9419476` also completed and archived separate
exact-reconstruction and text-generation panels for both modalities under
`tvl_flextok/logs/runs/flextok_canonical_generation_visual_smoke/`. The
one-update outputs are intentionally untrained and are only an end-to-end
artifact-path check, not a quality result.

Eight-sample generation memorization job `9420033` completed on one GPU in
33 minutes with exit code `0:0`.
It freezes the strongest 64-register eight-sample alignment checkpoint
(`flextok_latent_overfit8_reg64_flowft3/checkpoint_best_joint.pth`) and trains
the shared text-conditioned discrete-register GPT for 2,000 one-batch epochs
on the same eight deterministic SSVTP validation examples. It uses an explicit
learning rate of `3e-4`, 50 warmup epochs, no text-condition dropout, and
top-1 deterministic visualization. The pre-v4 checkpoint loader permits only
the two absent learned-null parameters, initialized to zero; every other state
mismatch remains fatal. Outputs stage under
`/scratch/peilinwu/tvl_flextok/runs/canonical_generation_overfit8_reg64/` and
archive to
`tvl_flextok/logs/runs/flextok_canonical_generation_overfit8_reg64/`.
The best checkpoint is epoch 1984 with 100% token accuracy for both modalities,
validation cross entropy `3.83e-7`, and perplexity `1.0000004`. Shuffled-caption
CE is 2.81 for vision and 13.48 for tactile. It is archived at:

```text
tvl_flextok/logs/runs/flextok_canonical_generation_overfit8_reg64/checkpoints/checkpoint_best_generation.pth
```

The original in-process panels are invalid because the launch predates a fix
to `generate()` that duplicated the first ID and omitted the last ID. Corrected
post-training visualization job `9420298` completed with exit code `0:0` and
saved admissible panels under `corrected_visualization/`. At `k=64`, generated
vision reaches 33.4059 dB / 0.90873 SSIM against an exact-register ceiling of
33.4059 / 0.90873; generated tactile reaches 37.5625 dB / 0.95650 against an
exact-register ceiling of 37.5621 / 0.95650. This proves exact memorization of
the eight register sequences, not held-out generation.

Corrected full-data alignment job `9419434` is the accepted replacement. It is
running on one GPU with 64 registers, all 38,830 SSVTP+HCT training pairs,
4,316 held-out validation pairs, batch size 8, accumulation 4, depth-8 flow
decoders, uniform per-sample `K in [1,64]`, and 100 epochs. Its scratch root is
`/scratch/peilinwu/tvl_flextok/runs/canonical_alignment_full/`; successful
completion archives best checkpoints and visual diagnostics to
`tvl_flextok/logs/runs/flextok_alignment_canonical_full_reg64_1gpu/`.
At epoch 3, interim validation retrieval top-1 is 82.38% and mean flow loss is
0.6572. Training remains active; these are not convergence metrics.

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

## Completed 64-Register Capacity Ablation

The capacity ablation tests whether the 32-register bottleneck limits
full-prefix detail. It trains a new 64-register tokenizer from scratch on the same eight
pairs while holding shared registers (8), FSQ levels, hidden size, resampler
depth, latent patch size, and depth-2 flow decoder fixed. The 224x224 input
produces a 28x28 VAE latent; patch size 2 exposes 196 latent memory tokens, so
the change reduces compression from about 6.1x to about 3.1x.

The initial joint run uses 1,000 updates at learning rate `3e-4` and saves
fixed-noise grids for `k={1,4,8,16,32,64}`. It is followed by three dependent
1,000-update reconstruction-only phases with the tokenizer frozen and fresh
cosine schedules at `1e-4`, `5e-5`, and `2.5e-5`.

| Phase | Slurm job | Result | Best validation flow loss |
| --- | ---: | --- | ---: |
| Joint 64-register alignment | 9348795 | Completed | 0.4160 |
| Flow continuation 1 | 9348870 | Completed | 0.2704 |
| Flow continuation 2 | 9348873 | Completed | 0.2230 |
| Flow continuation 3 | 9348876 | Completed | **0.2012** |

All jobs completed with exit code `0:0`. Each continuation used Slurm
`afterok`, warm-started the preceding archived best joint checkpoint, and
retained the same 64-register architecture. The complete experiment contains
4,000 updates, and all outputs were archived under:

```text
tvl_flextok/logs/runs/flextok_latent_overfit8_reg64*/
```

The authoritative checkpoint is phase-3 epoch 991:

```text
tvl_flextok/logs/runs/flextok_latent_overfit8_reg64_flowft3/checkpoints/checkpoint_best_joint.pth
```

It is a format-version-3 compact checkpoint of about 121 MiB with 64 total
registers, 8 shared registers, a depth-2 flow decoder, and frozen tokenizer
weights. Its validation flow loss is 0.2012 (vision 0.3021, tactile 0.1003).

At epoch 900, fixed-noise full-prefix reconstruction is:

| Registers | Vision PSNR / SSIM | Tactile PSNR / SSIM |
| ---: | ---: | ---: |
| 32-register baseline, `k=32` | 27.62 dB / 0.9841 | 33.98 dB / 0.9947 |
| 64-register model, `k=32` | 31.96 dB / 0.9942 | 37.16 dB / 0.9975 |
| 64-register model, `k=64` | **31.99 dB / 0.9942** | **37.60 dB / 0.9977** |

Most of the vision gain appears by `k=16`; `k=64` adds little over `k=32` for
vision, while tactile still benefits measurably. The 64-register tokenizer has
87.5% eight-pair retrieval rather than the baseline's 100%, and only about
6.6%/6.8% discrete-code utilization for vision/tactile. It is therefore the
best reconstruction checkpoint, but not an unqualified replacement for the
32-register alignment checkpoint. The full loss curve is:

```text
tvl_flextok/logs/runs/flextok_latent_overfit8_reg64_flowft3/loss_curve_full_history.png
```

## Historical Autoregressive Latent Reconstruction Ablation

This removed experimental stage froze the authoritative 64-register tokenizer
and predicts the same modality's continuous VAE latent. A causal Transformer
cross-attends to a randomly selected quantized register prefix and
autoregresses 196 raster-ordered `2x2` latent patches. Each next patch is
trained with diagonal-Gaussian NLL; free-running mean predictions are decoded
by the frozen VAE for prefix diagnostics.

GPU smoke job `9402699` completed with exit code `0:0`. It verified real
dataset loading, alignment-checkpoint restoration, one forward/backward
update, validation at `k={1,2,4,8,16,32,64}`, and format-version-4 compact
checkpoint writing.

Matched eight-pair, 1,000-update ablations were submitted with identical
width/depth, optimization, prefix sampling, and visualization settings:

| Variant | Slurm job | Decoder weights | Status at submission |
| --- | ---: | --- | --- |
| Separate | 9402722 | Independent vision/tactile Transformers | Completed; invalid objective |
| Shared | 9402723 | One modality-conditioned Transformer | Completed; invalid objective |

Both runs use the same eight deterministic SSVTP validation pairs as training
and validation. They test architecture and memorization, not held-out
generalization. Successful jobs archive to:

```text
tvl_flextok/logs/runs/flextok_latent_ar_separate_overfit8/
tvl_flextok/logs/runs/flextok_latent_ar_shared_overfit8/
```

## Full-Dataset Alignment Run

Four-GPU Slurm job `9402814` never started and was cancelled while pending.
It was replaced by one-GPU job `9413754`, which attempted to train a new 64-register
tokenizer without an overfit subset. It uses all materialized SSVTP and HCT
splits: 38,830 training pairs and 4,316 held-out validation pairs. The run uses
batch size 8 with 4 gradient-accumulation steps (effective batch 32), 100
epochs, 8 shared registers, depth-8 modality flow decoders, and prefix budgets
`k={1,2,4,8,16,32,64}`.

The run stages checkpoints under `/scratch/$USER/tvl_flextok/` and will
archive best flow, retrieval, and joint checkpoints plus reconstruction grids
to:

```text
tvl_flextok/logs/runs/flextok_alignment_full_reg64_1gpu/
```

Job `9413754` was cancelled and superseded by the canonical correction above;
it produced no accepted result.
