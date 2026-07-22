# TVL-FlexTok Model Audit and Multimodal Fusion Review

Date: 2026-07-22

## Executive Conclusion

TVL-FlexTok is now a discrete, ordered multimodal tokenizer rather than a
pooled-feature alignment module. Frozen TVL patch sequences are compressed by
strictly causal register resamplers, quantized with FSQ, and decoded through
modality-specific rectified-flow models in frozen VAE latent space. A separate
causal Transformer models the discrete register sequence.

`tvl_llama` is not the intended consumer and is outside this package's defect
list. Future consumers should use the documented register-code interface.

## Decisions

| Component | Decision | Reason |
| --- | --- | --- |
| Frozen TVL vision/tactile encoders | Keep | Patch sequences provide useful pretrained semantics without destabilizing tokenizer training. |
| Register resampler | Causal self-attention plus feature cross-attention | Immutable feature memory prevents suffix information from leaking into prefixes. |
| Nested dropout | Keep and pair across modalities | It trains every sampled prefix as a valid information budget. |
| FSQ | Use `[8,8,8,5,5,5]` | It produces discrete language-model-compatible codes without codebook collapse. |
| Leading alignment registers | Keep as retrieval head | Global contrastive supervision preserves paired semantic correspondence. |
| Private/preservation loss | Remove | It imposes no justified definition of task-relevant modality-specific information. |
| TVL feature reconstruction | Remove | Registers should compact and discard unnecessary frozen-encoder details. |
| Embedding flow matching | Remove | Transport between trainable pooled embeddings is not FlexTok reconstruction and admits collapse. |
| VAE-latent rectified flow | Always train in alignment | It directly tests and trains recoverability of original modality inputs from register prefixes. |
| Parallel image-query decoder | Remove from training | Causal query attention did not make simultaneous pixel prediction autoregressive. |
| Generation stage | Predict target FSQ register IDs | This is a genuine next-token objective and matches the interface future multimodal models consume. |

## Losses

### Global cross-modal contrastive loss

The leading active registers are pooled and projected into normalized vision
and tactile embeddings. Symmetric InfoNCE uses negatives gathered from every
DDP rank. It preserves sample-level shared semantics but does not force every
register position to match across modalities.

### Rectified-flow reconstruction loss

For each modality, the frozen VAE produces target latent `x1`; Gaussian noise
is `x0`; and training samples `xt = x0 + t(x1-x0)`. The modality flow decoder
predicts velocity `x1-x0`, conditioned on the retained FSQ register prefix.
This gives every retained register a reconstruction gradient without requiring
it to reproduce frozen TVL features.

### Autoregressive code loss

The generation stage minimizes next-token cross entropy over FSQ IDs. The
sequence has fixed length `R` and no EOS class. A single shared GPT is
conditioned on contextual OpenCLIP caption tokens and a vision/tactile modality
BOS. It is teacher forced with target IDs shifted right during training,
causally masked, and autoregressive over its own previously generated IDs at
inference. The GPT does not consume registers from the opposite modality.

After generation, each FSQ ID is mapped back to its quantized register
embedding. A selected register prefix conditions the frozen modality flow
decoder, which produces a VAE latent from noise. The frozen VAE decoder sees
only that latent.

## Diagnostics and Remaining Risks

- Fixed-noise reconstructions at `k={1,4,8,16,32}` distinguish token evidence
  from stochastic decoder variation.
- Track flow loss, PSNR, SSIM, optional LPIPS, prefix monotonicity, retrieval,
  register variance, and FSQ utilization. Image plausibility alone is not proof
  that omitted details were encoded.
- The shared pretrained VAE was trained on natural images. Its tactile-domain
  reconstruction ceiling must be measured before attributing tactile errors to
  register compression.
- HCT should use episode-level rather than frame-level splits before reporting
  final generalization metrics.
- The current text-conditioned GPT is a representation probe. A future LLM/VLA
  may predict the same FSQ IDs from language, state, action, or temporal
  observation context.
- Flow sampling cost grows with integration steps; report quality and latency
  together.

## Current Empirical Status

The completed alignment implementation diagnostic uses eight distinct static SSVTP
validation pairs for both training and evaluation. It is a memorization test,
not evidence of held-out or trajectory-level generalization. After 300 joint
alignment updates and three 1,000-update flow-only continuations with a frozen
tokenizer, best validation flow loss reached 0.3075.

At the latest saved visualization point, the full 32-register prefix reached
27.62 dB PSNR and 0.9841 SSIM for vision, and 33.98 dB and 0.9947 for tactile.
Tactile reconstruction is monotonic over the evaluated prefixes. Vision is
not strictly monotonic because `k=4` is slightly worse than `k=1`, and visible
latent-grid texture remains. These results verify register-conditioned
reconstruction and noncollapsed codes on the memorized subset, while leaving
full-data generalization unresolved.

The discrete generation memorization run `9420033` froze the strongest
64-register eight-sample alignment checkpoint and trained a 66.8M-parameter
shared Register GPT for 2,000 one-batch epochs. It completed with 100% token
accuracy for both modalities and validation cross entropy `3.84e-7` on the
same eight examples. Shuffled-caption CE was 2.81 for vision and 13.48 for
tactile, confirming that the memorized predictor uses text. This is not
held-out generation evidence. Its in-process image panels used a subsequently
fixed off-by-one generation loop and remain invalid. Corrected post-training
visualization reproduced the exact-register flow ceiling at `k=64`: vision
33.4059 dB / 0.90873 SSIM and tactile 37.5625 dB / 0.95650 SSIM.

Corrected full-data alignment job `9419434` remains active. At epoch 3 its
held-out retrieval top-1 was 82.38% and mean flow loss was 0.6572. These are
interim metrics, not convergence claims.

Exact configuration, run history, checkpoints, plots, and reconstruction paths
are maintained in [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).

## LLM and VLA Integration Guidance

The transferable interface is a variable-length sequence of discrete register
IDs plus modality, position, and validity metadata. Useful precedents include:

| Work | Relevant lesson |
| --- | --- |
| [FlexTok](https://arxiv.org/abs/2502.13967) | Ordered FSQ registers, nested dropout, and flow decoding make token budget an explicit coarse-to-fine control. |
| [Flamingo](https://arxiv.org/abs/2204.14198) | A fixed latent resampler and gated cross-attention can connect frozen encoders to language models. |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | Learned queries benefit from multiple pretraining signals before LLM integration. |
| [LLaVA](https://arxiv.org/abs/2304.08485) | Uncompressed patch projection is an essential information-retention baseline. |
| [TokenLearner](https://arxiv.org/abs/2106.11297) | Adaptive token selection can preserve supervised regions, but token order is not automatic. |
| [MM1](https://arxiv.org/abs/2403.09611) | Token count and input resolution can matter more than connector complexity. |
| [Matryoshka Multimodal Models](https://arxiv.org/abs/2405.17430) | Train the downstream objective at every supported token budget. |
| [pi0](https://arxiv.org/abs/2410.24164) | Continuous action flow should operate in action space and condition on state/history, not reuse embedding flow. |

For an LLM, compare discrete register embeddings against full patch projection
at matched context budgets. For a VLA, add temporal tactile windows,
proprioception, previous actions, and contact state; use register codes as
observation context for a separate action head or action-flow expert.
