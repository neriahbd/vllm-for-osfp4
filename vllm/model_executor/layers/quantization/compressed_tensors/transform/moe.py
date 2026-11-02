# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loading and grouped input scaling for independent expert diagonals."""

from dataclasses import dataclass

import torch
from compressed_tensors.transform import TransformConfig
from compressed_tensors.utils import match_name


@dataclass(frozen=True)
class ExpertTransform:
    projection: str
    expert_id: int
    inverse: bool
    precision: torch.dtype


def resolve_moe_input_transforms(
    prefix: str,
    config: TransformConfig | None,
    num_experts: int,
    projection_names: dict[str, str],
) -> dict[str, ExpertTransform]:
    """Resolve original expert projection names before vLLM fuses their weights."""
    bindings: dict[str, ExpertTransform] = {}
    if config is None:
        return bindings
    for expert_id in range(num_experts):
        for projection_name, projection in projection_names.items():
            name = f"{prefix}.{expert_id}.{projection_name}"
            for group_name, scheme in config.config_groups.items():
                if scheme.type != "diagonal":
                    continue
                for args in scheme.apply:
                    if not args.is_online() or not (
                        any(t == "Linear" or match_name(name, t) for t in args.targets)
                        and not any(
                            t == "Linear" or match_name(name, t) for t in args.ignore
                        )
                    ):
                        continue
                    if args.location != "input" or scheme.head_dim is not None:
                        raise NotImplementedError(
                            "MoE supports full-width input diagonals"
                        )
                    key = f"{name}.{group_name}_input.weight"
                    if any(k.startswith(name + ".") for k in bindings):
                        raise ValueError(f"Multiple input transforms target {name}")
                    bindings[key] = ExpertTransform(
                        projection, expert_id, args.inverse, scheme.precision
                    )
    for expert_id in range(num_experts):
        selected = [
            b
            for b in bindings.values()
            if b.expert_id == expert_id and b.projection == "w13"
        ]
        expected = list(projection_names.values()).count("w13")
        if selected and (len(selected) != expected or len(set(selected)) != 1):
            raise ValueError(
                "Fused gate/up projections require matching input transforms"
            )
    return bindings


class ExpertDiagonal(torch.nn.Module):
    """Apply each expert's vector to its contiguous group of routed rows."""

    def __init__(self, num_experts: int, width: int, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(num_experts, width, dtype=dtype), requires_grad=False
        )
        self.register_buffer(
            "inverse", torch.zeros(num_experts, dtype=torch.bool), persistent=False
        )

    def forward(self, inputs: torch.Tensor, expert_offsets: torch.Tensor):
        rows = torch.arange(inputs.shape[0], device=inputs.device)
        experts = torch.bucketize(rows, expert_offsets[1:].contiguous(), right=True)
        valid = rows < expert_offsets[-1]
        experts = experts.clamp_max(self.weight.shape[0] - 1)
        scale = self.weight[experts, : inputs.shape[-1]].to(inputs)
        transformed = torch.where(
            self.inverse[experts, None], inputs / scale, inputs * scale
        )
        return torch.where(valid[:, None], transformed, torch.zeros_like(inputs))


class MoEInputTransforms(torch.nn.Module):
    """Own checkpoint vectors independently of packed weights and global scales."""

    def __init__(
        self,
        bindings: dict[str, ExpertTransform],
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_rank=0,
        tp_size=1,
        full_hidden_size: int | None = None,
        full_intermediate_size: int | None = None,
        tp_shard_with_padding: bool = False,
    ):
        super().__init__()
        self.bindings = bindings
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.tp_shard_with_padding = tp_shard_with_padding
        self.full_widths = {
            "w13": full_hidden_size or hidden_size,
            "w2": full_intermediate_size or intermediate_size * tp_size,
        }
        self.loaded: set[str] = set()
        self.loaded_rows: set[tuple[str, int]] = set()
        for projection, width in (("w13", hidden_size), ("w2", intermediate_size)):
            dtypes = {
                b.precision for b in bindings.values() if b.projection == projection
            }
            if len(dtypes) > 1:
                raise ValueError(
                    "Each fused projection requires one transform precision"
                )
            self.add_module(
                projection,
                ExpertDiagonal(num_experts, width, next(iter(dtypes), torch.bfloat16)),
            )
            if not dtypes:
                transform = getattr(self, projection)
                weight = transform.weight.detach()
                del transform.weight
                transform.register_buffer("weight", weight, persistent=False)

    def load_checkpoint_weight(self, name: str, value: torch.Tensor) -> str | None:
        binding = self.bindings.get(name)
        if binding is None:
            return None
        if value.ndim != 1 or value.dtype != binding.precision:
            raise ValueError(f"Invalid diagonal vector for {name}")
        if value.numel() != self.full_widths[binding.projection]:
            raise ValueError(f"Invalid diagonal width for {name}")
        transform = getattr(self, binding.projection)
        destination = transform.weight[binding.expert_id]
        if binding.projection == "w2":
            if self.tp_shard_with_padding:
                start = self.tp_rank * destination.numel()
                value = value[start : start + destination.numel()]
            else:
                if value.numel() % self.tp_size:
                    raise ValueError(f"Cannot shard diagonal vector {name}")
                value = value.chunk(self.tp_size)[self.tp_rank]
        if value.numel() > destination.numel():
            raise ValueError(f"Diagonal vector is wider than projection: {name}")
        value = value.to(destination)
        row = (binding.projection, binding.expert_id)
        if row in self.loaded_rows:
            if not torch.equal(destination[: value.numel()], value):
                raise ValueError(f"Fused gate/up diagonal vectors differ: {name}")
        else:
            destination.data[: value.numel()].copy_(value)
            transform.inverse[binding.expert_id] = binding.inverse
            self.loaded_rows.add(row)
        self.loaded.add(name)
        return f"input_transforms.{binding.projection}.weight"

    def validate_loaded(self):
        missing = self.bindings.keys() - self.loaded
        if missing:
            raise ValueError(f"Missing expert diagonal weights: {sorted(missing)}")
