# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify expert checkpoint identity, TP slicing, and routed diagonal arithmetic."""

import pytest
import torch
from compressed_tensors.transform import TransformConfig

from vllm.model_executor.layers.quantization.compressed_tensors.transform.moe import (
    MoEInputTransforms,
    resolve_moe_input_transforms,
)


def make_transforms(tp_rank=0, tp_size=1, inverse=False):
    config = TransformConfig.model_validate(
        {
            "config_groups": {
                "smooth_quant": {
                    "type": "diagonal",
                    "precision": "torch.bfloat16",
                    "apply": [
                        {
                            "location": "input",
                            "inverse": inverse,
                            "targets": ["re:model.experts.*"],
                            "ignore": [],
                        }
                    ],
                }
            }
        }
    )
    bindings = resolve_moe_input_transforms(
        "model.experts",
        config,
        3,
        {"gate_proj": "w13", "up_proj": "w13", "down_proj": "w2"},
    )
    return MoEInputTransforms(bindings, 3, 4, 8 // tp_size, tp_rank, tp_size)


def load_vectors(transforms):
    for name, binding in transforms.bindings.items():
        width = 8 if binding.projection == "w2" else 4
        vector = torch.arange(1, width + 1, dtype=torch.bfloat16) + binding.expert_id
        transforms.load_checkpoint_weight(name, vector)
    transforms.validate_loaded()


@pytest.mark.parametrize("inverse", [False, True])
def test_grouped_rows_use_their_expert_and_ignore_padding(inverse):
    transforms = make_transforms(inverse=inverse)
    load_vectors(transforms)
    # Expert 1 receives no rows; final two rows are padding.
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    inputs = torch.arange(28, dtype=torch.bfloat16).reshape(7, 4)
    original = inputs.clone()
    output = transforms.w13(inputs, offsets)
    expected = torch.zeros_like(inputs)
    for expert, start, end in ((0, 0, 2), (2, 2, 5)):
        scale = transforms.w13.weight[expert]
        expected[start:end] = (
            inputs[start:end] / scale if inverse else inputs[start:end] * scale
        )
    assert torch.equal(output, expected)
    assert torch.equal(inputs, original)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_tp_replicates_gate_up_and_shards_down(tp_rank):
    transforms = make_transforms(tp_rank, 2)
    load_vectors(transforms)
    for expert in range(3):
        assert torch.equal(
            transforms.w13.weight[expert],
            torch.arange(1, 5, dtype=torch.bfloat16) + expert,
        )
        expected = (torch.arange(1, 9, dtype=torch.bfloat16) + expert).chunk(2)[tp_rank]
        assert torch.equal(transforms.w2.weight[expert], expected)


def test_missing_checkpoint_vector_is_not_silently_identity():
    transforms = make_transforms()
    with pytest.raises(ValueError, match="Missing expert diagonal"):
        transforms.validate_loaded()


def test_fused_gate_up_rejects_different_vectors():
    transforms = make_transforms()
    transforms.load_checkpoint_weight(
        "model.experts.0.gate_proj.smooth_quant_input.weight",
        torch.ones(4, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="vectors differ"):
        transforms.load_checkpoint_weight(
            "model.experts.0.up_proj.smooth_quant_input.weight",
            torch.full((4,), 2, dtype=torch.bfloat16),
        )


def test_partial_gate_up_target_is_rejected():
    config = TransformConfig.model_validate(
        {
            "config_groups": {
                "smooth_quant": {
                    "type": "diagonal",
                    "apply": [
                        {"location": "input", "targets": ["model.experts.0.gate_proj"]}
                    ],
                }
            }
        }
    )
    with pytest.raises(ValueError, match="matching input transforms"):
        resolve_moe_input_transforms(
            "model.experts",
            config,
            1,
            {"gate_proj": "w13", "up_proj": "w13", "down_proj": "w2"},
        )


def test_save_reload_preserves_diagonal_bytes(tmp_path):
    from safetensors.torch import load_file, save_file

    transforms = make_transforms()
    load_vectors(transforms)
    tensors = {}
    for name, binding in transforms.bindings.items():
        tensors[name] = (
            getattr(transforms, binding.projection).weight[binding.expert_id].clone()
        )
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    restored = make_transforms()
    for name, tensor in load_file(path).items():
        restored.load_checkpoint_weight(name, tensor)
    restored.validate_loaded()
    for projection in ("w13", "w2"):
        assert torch.equal(
            getattr(restored, projection).weight.view(torch.uint8),
            getattr(transforms, projection).weight.view(torch.uint8),
        )


@pytest.mark.parametrize("width", [0, 3, 5])
def test_wrong_width_cannot_be_hidden_by_padding(width):
    transforms = make_transforms()
    with pytest.raises(ValueError, match="Invalid diagonal width"):
        transforms.load_checkpoint_weight(
            "model.experts.0.gate_proj.smooth_quant_input.weight",
            torch.ones(width, dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("scheme_name", ["NVFP4", "NVFP4A16"])
def test_native_checkpoint_fixture_preserves_per_expert_metadata(tmp_path, scheme_name):
    import json

    from safetensors.torch import load_file

    from tests.quantization.test_moe_diagonal_loading import make_checkpoints

    original, scaled = make_checkpoints(tmp_path, scheme_name)
    baseline, transformed = (
        load_file(original / "model.safetensors"),
        load_file(scaled / "model.safetensors"),
    )
    config = json.loads((scaled / "config.json").read_text())
    tfm_config = TransformConfig.model_validate(
        config["quantization_config"]["transform_config"]
    )
    bindings = resolve_moe_input_transforms(
        "model.layers.0.mlp.experts",
        tfm_config,
        4,
        {"gate_proj": "w13", "up_proj": "w13", "down_proj": "w2"},
    )
    transforms = MoEInputTransforms(bindings, 4, 256, 512)
    for name in bindings:
        transforms.load_checkpoint_weight(name, transformed[name])
        projection = name.removesuffix(".smooth_quant_input.weight")
        assert torch.equal(
            baseline[projection + ".weight_packed"],
            transformed[projection + ".weight_packed"],
        )
        multiplier = transformed[name][0].float()
        assert torch.equal(
            transformed[projection + ".weight_scale"].float() * multiplier,
            baseline[projection + ".weight_scale"].float(),
        )
        if scheme_name == "NVFP4":
            assert torch.equal(
                transformed[projection + ".input_global_scale"] * multiplier,
                baseline[projection + ".input_global_scale"],
            )
        else:
            assert projection + ".input_global_scale" not in transformed
    transforms.validate_loaded()


def test_ungated_projection_aliases_and_untargeted_identity():
    config = TransformConfig.model_validate(
        {
            "config_groups": {
                "smooth_quant": {
                    "type": "diagonal",
                    "precision": "torch.bfloat16",
                    "apply": [{"location": "input", "targets": ["model.experts.1.w1"]}],
                }
            }
        }
    )
    bindings = resolve_moe_input_transforms(
        "model.experts", config, 3, {"w1": "w13", "w2": "w2"}
    )
    transforms = MoEInputTransforms(bindings, 3, 4, 8)
    transforms.load_checkpoint_weight(
        "model.experts.1.w1.smooth_quant_input.weight",
        torch.full((4,), 2, dtype=torch.bfloat16),
    )
    transforms.validate_loaded()
    assert "w2.weight" not in dict(transforms.named_parameters())
    inputs = torch.ones(3, 4, dtype=torch.bfloat16)
    assert torch.equal(
        transforms.w13(inputs, torch.arange(4)),
        torch.tensor([[1] * 4, [2] * 4, [1] * 4], dtype=torch.bfloat16),
    )


def test_padded_tp_channels_are_identity():
    transforms = make_transforms()
    padded = MoEInputTransforms(
        transforms.bindings,
        3,
        8,
        8,
        tp_rank=1,
        tp_size=2,
        full_hidden_size=4,
        full_intermediate_size=8,
    )
    load_vectors(padded)
    assert torch.equal(padded.w13.weight[:, 4:], torch.ones(3, 4, dtype=torch.bfloat16))
    assert torch.equal(padded.w2.weight[:, 4:], torch.ones(3, 4, dtype=torch.bfloat16))


def test_diagonal_resolver_leaves_other_transform_types_to_existing_paths():
    config = TransformConfig.model_validate(
        {
            "config_groups": {
                "v": {
                    "type": "hadamard",
                    "apply": [{"location": "input", "targets": ["Linear"]}],
                }
            }
        }
    )
    assert (
        resolve_moe_input_transforms(
            "model.experts",
            config,
            2,
            {"gate_proj": "w13", "up_proj": "w13", "down_proj": "w2"},
        )
        == {}
    )


def test_down_vector_uses_the_same_padded_tp_stride_as_weights():
    bindings = make_transforms().bindings
    transforms = MoEInputTransforms(
        bindings,
        3,
        4,
        6,
        tp_rank=1,
        tp_size=2,
        full_intermediate_size=8,
        tp_shard_with_padding=True,
    )
    load_vectors(transforms)
    for expert in range(3):
        expected = torch.ones(6, dtype=torch.bfloat16)
        expected[:2] = torch.tensor([7, 8], dtype=torch.bfloat16) + expert
        assert torch.equal(transforms.w2.weight[expert], expected)
