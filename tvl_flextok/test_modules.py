"""CPU tests for the TVL-FlexTok tokenizer and generative heads."""

import os
import sys
import tempfile
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvl_flextok.losses.alignment_loss import CrossModalAlignmentLoss
from tvl_flextok.models.autoregressive_decoder import AutoregressiveDecoder
from tvl_flextok.models.cross_modal_alignment import CrossModalAlignmentModel
from tvl_flextok.models.flow_decoder import LatentFlowDecoder
from tvl_flextok.models.fsq import FiniteScalarQuantizer
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
    assert torch.allclose(before, after, atol=1e-6), "suffix registers leaked into an earlier prefix"


def test_paired_nested_dropout_and_outputs():
    model = CrossModalAlignmentModel(
        modality_configs={
            "vision": {"input_dim": 24, "feature_type": "sequence"},
            "tactile": {"input_dim": 16, "feature_type": "sequence"},
        },
        hidden_dim=32, n_registers=8, n_shared=2, n_layers=2, n_heads=4,
        dropout=0.0, fsq_levels=(4, 4, 3),
    ).train()
    output = model({"vision": torch.randn(3, 5, 24), "tactile": torch.randn(3, 6, 16)})
    assert output["vision_k_keep"] == output["tactile_k_keep"] == output["k_keep"]
    for modality in ("vision", "tactile"):
        assert output[f"{modality}_all_tokens_full"].shape == (3, 8, 32)
        assert output[f"{modality}_code_ids"].shape == (3, 8)
        assert output[f"{modality}_shared"].shape == (3, 32)
    loss = CrossModalAlignmentLoss(["vision", "tactile"])(output)["total_loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


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
        modality_configs={
            "vision": {
                "input_dim": 24, "feature_type": "sequence",
                "latent_channels": 4, "latent_patch_size": 2,
            },
        },
        hidden_dim=32, n_registers=4, n_shared=2, n_layers=1, n_heads=4,
        dropout=0.0, nested_dropout=False, fsq_levels=(4, 4, 3),
        tokenizer_input="vae_tvl",
    ).eval()
    features = {"vision": torch.randn(2, 5, 24)}
    latents = {"vision": torch.randn(2, 4, 8, 8)}
    output = model(features, latent_dict=latents)
    changed = model(features, latent_dict={"vision": latents["vision"] + 2.0})
    assert output["vision_continuous_tokens"].shape == (2, 4, 32)
    assert not torch.allclose(
        output["vision_continuous_tokens"], changed["vision_continuous_tokens"]
    )
    output["vision_continuous_tokens"].square().mean().backward()
    projector = model.latent_projectors["vision"]
    assert projector.weight.grad is not None and projector.weight.grad.abs().sum() > 0


def test_latent_flow_decoder():
    decoder = LatentFlowDecoder(
        register_dim=32, hidden_dim=32, depth=2, n_heads=4, patch_size=2,
    )
    latents = torch.randn(2, 4, 8, 8)
    registers = torch.randn(2, 4, 32, requires_grad=True)
    velocity = decoder(latents, torch.rand(2), registers)
    assert velocity.shape == latents.shape
    loss = decoder.flow_loss(latents, registers, condition_dropout=0.5)
    loss.backward()
    assert registers.grad is not None and registers.grad.abs().sum() > 0
    with torch.no_grad():
        sample = decoder.sample(
            registers.detach(), (4, 8, 8), steps=2, guidance_scale=2.0
        )
    assert sample.shape == latents.shape


def test_autoregressive_register_decoder():
    model = AutoregressiveDecoder(
        vocab_size=48, register_dim=32, hidden_dim=32,
        max_registers=8, depth=2, n_heads=4, dropout=0.0,
    ).eval()
    source = torch.randn(2, 8, 32)
    codes = torch.randint(0, 48, (2, 6))
    modality = torch.tensor([0, 1])
    logits, targets = model(source, codes, modality)
    assert logits.shape == (2, 7, 49)
    assert targets.shape == (2, 7)
    changed = codes.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % 48
    changed_logits, _ = model(source, changed, modality)
    assert torch.allclose(logits[:, :4], changed_logits[:, :4], atol=1e-6)
    model.loss(source, codes, modality).backward()
    assert any(p.grad is not None for p in model.parameters())
    generated = model.generate(source, modality, max_length=3, top_k=4)
    assert generated.shape[0] == 2 and generated.shape[1] <= 3


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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        print(f"{test.__name__} ...", end=" ", flush=True)
        test()
        print("ok")
    print(f"{len(tests)} tests passed")
