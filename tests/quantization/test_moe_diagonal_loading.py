# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native GPU loading regression for both compressed-tensors NVFP4 MoE formats."""

import copy

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from safetensors.torch import save_file

from tests.models.utils import check_logprobs_close


def make_checkpoints(tmp_path, scheme_name):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    state = {
        n: p.contiguous() for n, p in model.state_dict().items() if ".experts." not in n
    }
    targets = []
    for expert in range(4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.0.mlp.experts.{expert}.{projection}"
            targets.append(name)
            out_width, in_width = (
                (256, 512) if projection == "down_proj" else (512, 256)
            )
            state[f"{name}.weight_packed"] = torch.randint(
                0, 256, (out_width, in_width // 2), dtype=torch.uint8
            )
            state[f"{name}.weight_scale"] = torch.full(
                (out_width, in_width // 16), 1 / 128, dtype=torch.float8_e4m3fn
            )
            state[f"{name}.weight_global_scale"] = torch.ones(1)
            if scheme_name == "NVFP4":
                # Different fixed activation scales expose accidental expert reduction.
                state[f"{name}.input_global_scale"] = torch.tensor([64.0 * (2**expert)])
            state[f"{name}.smooth_quant_input.weight"] = torch.ones(
                in_width, dtype=torch.bfloat16
            )
    quant_scheme = preset_name_to_scheme(scheme_name, targets)
    config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {"group_0": quant_scheme.model_dump(mode="json")},
        "transform_config": {
            "config_groups": {
                "smooth_quant": {
                    "type": "diagonal",
                    "precision": "torch.bfloat16",
                    "apply": [{"location": "input", "targets": targets}],
                }
            }
        },
    }
    original = tmp_path / "identity"
    original.mkdir()
    config.save_pretrained(original)
    save_file(state, original / "model.safetensors", metadata={"format": "pt"})
    transformed = copy.deepcopy(state)
    for expert in range(4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.0.mlp.experts.{expert}.{projection}"
            multiplier = 2.0 ** (
                expert - 1 if projection != "down_proj" else 2 - expert
            )
            transformed[f"{name}.smooth_quant_input.weight"].fill_(multiplier)
            transformed[f"{name}.weight_scale"] = (
                state[f"{name}.weight_scale"].float() / multiplier
            ).to(torch.float8_e4m3fn)
            if scheme_name == "NVFP4":
                transformed[f"{name}.input_global_scale"] /= multiplier
    scaled = tmp_path / "scaled"
    scaled.mkdir()
    config.save_pretrained(scaled)
    save_file(transformed, scaled / "model.safetensors", metadata={"format": "pt"})
    return original, scaled


@pytest.mark.parametrize("scheme_name", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_moe_diagonal_checkpoint_preserves_generation(
    vllm_runner, tmp_path, scheme_name, tp_size, enforce_eager
):
    paths = make_checkpoints(tmp_path, scheme_name)
    results = []
    for path in paths:
        with vllm_runner(
            str(path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            enforce_eager=enforce_eager,
            max_model_len=128,
            max_num_seqs=2,
            tensor_parallel_size=tp_size,
            enable_prefix_caching=False,
            moe_backend="cutlass" if scheme_name == "NVFP4" else "humming",
        ) as llm:
            results.append(
                llm.generate_greedy_logprobs([[1, 4, 7, 8], [1, 6, 5]], 4, 5)
            )
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="identity diagonals",
        name_1="balanced expert diagonals",
    )
