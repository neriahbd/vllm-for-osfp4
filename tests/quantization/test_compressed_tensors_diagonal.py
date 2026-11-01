# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise diagonal checkpoint loading through native compressed-tensors layers."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.platforms import current_platform


def diagonal_config(targets=None, *, inverse=False, location="input", head_dim=None):
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "dense",
            "config_groups": {},
            "ignore": [],
            "transform_config": {
                "config_groups": {
                    "osfp4": {
                        "type": "diagonal",
                        "precision": "torch.bfloat16",
                        "head_dim": head_dim,
                        "apply": [
                            {
                                "targets": targets or ["projection"],
                                "location": location,
                                "inverse": inverse,
                            }
                        ],
                    }
                }
            },
        }
    )


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("parallel", ["row", "column"])
def test_diagonal_checkpoint_preserves_balanced_linear(
    dist_init, tmp_path, dtype, parallel
):
    """Powers of two make checkpoint balancing exact in every tested dtype."""
    _check_balanced_linear(tmp_path, dtype, parallel)


def _check_balanced_linear(tmp_path, dtype, parallel):
    rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    config = diagonal_config()
    layer_class = RowParallelLinear if parallel == "row" else ColumnParallelLinear
    kwargs = {"reduce_results": False} if parallel == "row" else {}
    device = torch.device(current_platform.device_type)
    with device:
        layer = layer_class(
            16,
            8,
            bias=False,
            params_dtype=dtype,
            quant_config=config,
            prefix="projection",
            **kwargs,
        )
    model = torch.nn.Module()
    model.add_module("projection", layer)
    weight = torch.arange(1, 129, dtype=torch.float32).reshape(8, 16).to(dtype) / 128
    scale = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.bfloat16).repeat(4)
    scale[8:] *= 2
    balanced = weight / scale.to(dtype)
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {"projection.weight": balanced, "projection.osfp4_input.weight": scale},
        checkpoint,
    )
    loaded = AutoWeightsLoader(model).load_weights(load_file(checkpoint).items())
    assert loaded == {"projection.weight", "projection.osfp4_input.weight"}

    inputs = (torch.arange(-48, 48, dtype=torch.float32).reshape(3, 32).to(dtype) / 32)[
        :, ::2
    ]
    if parallel == "row":
        columns = slice(rank * 16 // tp_size, (rank + 1) * 16 // tp_size)
        inputs, weight, balanced = (
            inputs[:, columns],
            weight[:, columns],
            balanced[:, columns],
        )
        scale = scale[columns]
    else:
        rows = slice(rank * 8 // tp_size, (rank + 1) * 8 // tp_size)
        weight, balanced = weight[rows], balanced[rows]
    assert_bytes(layer.osfp4_input.weight, scale)
    assert_bytes(layer.weight, balanced)
    layer.quant_method.process_weights_after_loading(layer)
    assert_bytes(layer.osfp4_input.weight, scale)
    assert layer.requires_unquantized_input
    with torch.no_grad():
        output, _ = layer(inputs.to(device))
    assert_bytes(
        output, torch.nn.functional.linear(inputs.to(device), weight.to(device))
    )


@pytest.mark.parametrize(
    "layer_class", [ColumnParallelLinear, RowParallelLinear, ReplicatedLinear]
)
def test_diagonal_input_transform_preserves_bias(dist_init, layer_class):
    """Input scaling must leave the Linear's bias contribution unchanged."""
    with torch.device(current_platform.device_type):
        layer = layer_class(
            16,
            8,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=diagonal_config(),
            prefix="projection",
        )
        weight = torch.arange(128, dtype=torch.bfloat16).reshape(8, 16) / 128
        bias = torch.arange(1, 9, dtype=torch.bfloat16) / 8
        scale = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.bfloat16).repeat(4)
        inputs = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16) / 16
    model = torch.nn.Module()
    model.add_module("projection", layer)
    AutoWeightsLoader(model).load_weights(
        {
            "projection.weight": weight / scale,
            "projection.bias": bias,
            "projection.osfp4_input.weight": scale,
        }.items()
    )
    layer.quant_method.process_weights_after_loading(layer)
    with torch.no_grad():
        output, _ = layer(inputs)
    assert_bytes(output, torch.nn.functional.linear(inputs, weight, bias))


def _check_tensor_parallel_worker(rank, init_file, folder):
    from pathlib import Path

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            distributed_init_method=f"file://{init_file}",
            local_rank=rank,
            backend="gloo",
        )
        initialize_model_parallel(2, 1, backend="gloo")
        rank_folder = Path(folder) / str(rank)
        rank_folder.mkdir()
        try:
            for parallel in ("row", "column"):
                _check_balanced_linear(rank_folder, torch.bfloat16, parallel)
        finally:
            cleanup_dist_env_and_memory()


def test_diagonal_checkpoint_tensor_parallel_loading(tmp_path, monkeypatch):
    """Load and execute both input-sharded and replicated scales on two CPU ranks."""
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    torch.multiprocessing.spawn(
        _check_tensor_parallel_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
    )


@pytest.mark.parametrize(
    "targets", [["mlp.gate_proj"], ["mlp.gate_proj", "mlp.up_proj"]]
)
def test_diagonal_rejects_independent_scales_on_fused_projection(dist_init, targets):
    config = diagonal_config(targets)
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    with (
        torch.device("cpu"),
        pytest.raises(NotImplementedError, match="fused projections"),
    ):
        MergedColumnParallelLinear(
            16, [8, 8], bias=False, quant_config=config, prefix="mlp.gate_up_proj"
        )


@pytest.mark.parametrize(
    "location,head_dim,inverse",
    [("output", None, False), ("input", 16, False), ("input", None, True)],
)
def test_diagonal_rejects_unsupported_transform_configurations(
    dist_init, location, head_dim, inverse
):
    config = diagonal_config(location=location, head_dim=head_dim, inverse=inverse)
    with (
        torch.device("cpu"),
        pytest.raises(NotImplementedError, match="full-width input"),
    ):
        ColumnParallelLinear(
            16, 8, bias=False, quant_config=config, prefix="projection"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_diagonal_compilation_preserves_activation_bytes(dist_init, dtype):
    with torch.device("cpu"):
        layer = ColumnParallelLinear(
            16, 8, bias=False, quant_config=diagonal_config(), prefix="projection"
        )
    scale = torch.linspace(0.5, 2.5, 16, dtype=torch.bfloat16)
    layer.osfp4_input.weight.weight_loader(layer.osfp4_input.weight, scale)
    compiled = torch.compile(
        layer.osfp4_input, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for batch_size in (3, 7):
        values = (
            torch.arange(batch_size * 32, dtype=torch.float32)
            .reshape(batch_size, 32)
            .to(dtype)[:, ::2]
        )
        with torch.no_grad():
            assert_bytes(compiled(values), layer.osfp4_input(values))
