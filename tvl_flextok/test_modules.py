"""CPU tests for the TVL-FlexTok tokenizer and generative heads."""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvl_flextok.losses.alignment_loss import CrossModalAlignmentLoss
from tvl_flextok.models.cross_modal_alignment import CrossModalAlignmentModel
from tvl_flextok.models.flow_decoder import LatentFlowDecoder
from tvl_flextok.models.fsq import FiniteScalarQuantizer
from tvl_flextok.models.register_gpt import TextConditionedRegisterGPT
from tvl_flextok.models.register_tokens import RegisterTokenModule


def test_fsq():
    quantizer = FiniteScalarQuantizer(32, (4, 4, 3))
    inputs = torch.randn(2, 7, 32, requires_grad=True)
    quantized, codes, indices = quantizer(inputs)
    assert quantized.shape == inputs.shape
    assert codes.shape == (2, 7)
    assert codes.max() < quantizer.vocab_size
    assert torch.equal(indices, quantizer.codes_to_indices(codes))
    assert torch.allclose(quantized.detach(), quantizer.codes_to_quantized(codes), atol=1e-6)
    quantized.square().mean().backward()
    assert inputs.grad is not None and inputs.grad.abs().sum() > 0


def test_register_prefix_is_causal():
    torch.manual_seed(0)
    module = RegisterTokenModule(
        input_dim=24, hidden_dim=32, n_registers=8, n_shared=2,
        n_layers=2, n_heads=4, dropout=0.0, nested_dropout=False,
        fsq_levels=(4, 4, 3),
    ).eval()
    features = torch.randn(2, 5, 24)
    with torch.no_grad():
        before = module(features, return_dict=True)["continuous"][:, :4].clone()
        module.register_tokens[:, 4:].normal_(mean=100, std=10)
        module.register_pos_embed[:, 4:].normal_(mean=-100, std=10)
        after = module(features, return_dict=True)["continuous"][:, :4]
    assert torch.allclose(before, after, atol=1e-6)


def test_per_sample_paired_nested_dropout():
    model = CrossModalAlignmentModel(
        modality_configs={
            "vision": {"input_dim": 24, "feature_type": "sequence"},
            "tactile": {"input_dim": 16, "feature_type": "sequence"},
        },
        hidden_dim=32, n_registers=8, n_shared=2, n_layers=1, n_heads=4,
        dropout=0.0, nested_dropout_mode="uniform", fsq_levels=(4, 4, 3),
    ).train()
    torch.manual_seed(7)
    output = model({"vision": torch.randn(16, 5, 24), "tactile": torch.randn(16, 6, 16)})
    assert output["k_keep"].shape == (16,)
    assert torch.equal(output["vision_k_keep"], output["tactile_k_keep"])
    assert output["k_keep"].unique().numel() > 1
    for modality in ("vision", "tactile"):
        expected = torch.arange(8)[None] < output["k_keep"][:, None]
        assert torch.equal(output[f"{modality}_active_mask"], expected)
        assert output[f"{modality}_all_tokens_full"].shape == (16, 8, 32)
    CrossModalAlignmentLoss(["vision", "tactile"])(output)["total_loss"].backward()


def test_pooled_mode_compatibility():
    model = CrossModalAlignmentModel(
        modality_configs={"vision": {"input_dim": 24, "feature_type": "pooled"}},
        hidden_dim=32, n_registers=4, n_shared=2, n_layers=1, n_heads=4,
        nested_dropout=False, fsq_levels=(4, 4, 3),
    ).eval()
    output = model({"vision": torch.randn(2, 24)})
    assert output["vision_all_tokens_full"].shape == (2, 4, 32)


def test_vae_latents_condition_register_encoder():
    model = CrossModalAlignmentModel(
        modality_configs={"vision": {
            "input_dim": 24, "feature_type": "sequence",
            "latent_channels": 4, "latent_patch_size": 2,
        }},
        hidden_dim=32, n_registers=4, n_shared=2, n_layers=1, n_heads=4,
        dropout=0.0, nested_dropout=False, fsq_levels=(4, 4, 3), tokenizer_input="vae_tvl",
    ).eval()
    features = {"vision": torch.randn(2, 5, 24)}
    latents = {"vision": torch.randn(2, 4, 8, 8)}
    output = model(features, latent_dict=latents)
    changed = model(features, latent_dict={"vision": latents["vision"] + 2.0})
    assert not torch.allclose(output["vision_continuous_tokens"], changed["vision_continuous_tokens"])


def test_latent_flow_decoder_masks_and_null_condition():
    decoder = LatentFlowDecoder(
        register_dim=32, hidden_dim=32, depth=2, n_heads=4, patch_size=2,
    ).eval()
    latents = torch.randn(2, 4, 8, 8)
    registers = torch.randn(2, 4, 32, requires_grad=True)
    padding = torch.tensor([[False, False, True, True], [False, False, False, True]])
    velocity = decoder(latents, torch.rand(2), registers, padding)
    assert velocity.shape == latents.shape and torch.isfinite(velocity).all()
    loss = decoder.flow_loss(
        latents, registers, condition_dropout=1.0, register_padding_mask=padding
    )
    loss.backward()
    assert decoder.null_register.grad is not None
    sample = decoder.sample(
        registers.detach(), (4, 8, 8), steps=2, guidance_scale=2.0,
        register_padding_mask=padding,
    )
    assert sample.shape == latents.shape and torch.isfinite(sample).all()


def test_text_conditioned_register_gpt():
    torch.manual_seed(0)
    model = TextConditionedRegisterGPT(
        vocab_size=48, text_dim=24, hidden_dim=32, max_registers=8,
        depth=2, n_heads=4, dropout=0.0,
    ).eval()
    text = torch.randn(2, 6, 24)
    text_mask = torch.tensor([[False] * 5 + [True], [False] * 4 + [True, True]])
    codes = torch.randint(48, (2, 8))
    modalities = torch.tensor([0, 1])
    logits = model(text, codes, modalities, text_mask)
    changed = codes.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 48
    changed_logits = model(text, changed, modalities, text_mask)
    assert logits.shape == (2, 8, 48)
    assert torch.allclose(logits[:, :5], changed_logits[:, :5], atol=1e-6)
    loss, accuracy, first = model.loss(text, codes, modalities, text_mask)
    assert torch.isfinite(loss + accuracy + first)
    assert 1.0 < float(loss) < 10.0
    loss.backward()
    generated = model.generate(text, modalities, text_mask, max_tokens=5, sample=False)
    assert generated.shape == (2, 5)
    expected = torch.empty(2, 0, dtype=torch.long)
    for _ in range(5):
        placeholder = torch.zeros(2, 1, dtype=torch.long)
        decoder_input = torch.cat([expected, placeholder], dim=1)
        next_code = model(text, decoder_input, modalities, text_mask)[:, -1].argmax(dim=-1)
        expected = torch.cat([expected, next_code[:, None]], dim=1)
    assert torch.equal(generated, expected)


def test_config_unknown_keys_fail():
    from tvl_flextok.train import get_args_parser, load_config_overrides
    args = get_args_parser().parse_args([])
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write("training:\n  base_lr: 0.001\n")
        path = handle.name
    try:
        try:
            load_config_overrides(args, path, ["test"])
        except KeyError as exc:
            assert "training.base_lr" in str(exc)
        else:
            raise AssertionError("unknown configuration key was accepted")
    finally:
        os.unlink(path)


def test_alignment_metadata_configures_generation():
    from tvl_flextok.train import apply_alignment_checkpoint_metadata, get_args_parser
    with tempfile.NamedTemporaryFile("wb", suffix=".pth", delete=False) as handle:
        path = handle.name
    torch.save({
        "stage": "alignment",
        "args": {
            "n_registers": 64, "n_shared": 8, "tokenizer_input": "vae_tvl",
            "flow_depth": 3, "flow_heads": 4,
        },
    }, path)
    try:
        args = get_args_parser().parse_args([
            "--stage", "generation", "--alignment_checkpoint", path,
        ])
        configured = apply_alignment_checkpoint_metadata(
            args, ["test", "--stage", "generation"]
        )
        assert configured.n_registers == 64
        assert configured.flow_depth == 3
        assert configured.flow_heads == 4
    finally:
        os.unlink(path)


def test_legacy_alignment_stack_allows_only_null_registers():
    from tvl_flextok.engine import load_alignment_stack

    alignment_model = torch.nn.Linear(4, 4)
    flow_decoders = torch.nn.ModuleDict({
        modality: LatentFlowDecoder(
            register_dim=8, hidden_dim=8, depth=1, n_heads=2, patch_size=2,
        )
        for modality in ("vision", "tactile")
    })
    legacy_flow = {
        key: value for key, value in flow_decoders.state_dict().items()
        if not key.endswith("null_register")
    }
    system = SimpleNamespace(
        tokenizer=SimpleNamespace(alignment_model=alignment_model),
        flow_decoders=flow_decoders,
    )
    load_alignment_stack(system, {
        "format_version": 3,
        "alignment_model": alignment_model.state_dict(),
        "flow_decoders": legacy_flow,
    })
    assert torch.equal(flow_decoders["vision"].null_register, torch.zeros(1, 1, 8))


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        print(f"{test.__name__} ...", end=" ", flush=True)
        test()
        print("ok")
