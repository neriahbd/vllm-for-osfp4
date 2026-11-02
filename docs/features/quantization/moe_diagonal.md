# MoE input diagonal transforms

Compressed-tensors MoE checkpoints can describe independent input diagonal vectors
for each expert using `transform_config`. The saved keys retain their original
projection names, for example
`model.layers.0.mlp.experts.0.down_proj.smooth_quant_input.weight`.
Weights must already contain inverse balancing. vLLM loads the vectors separately
and applies them to routed activations; it does not rebalance weights.

The initial backend paths are native vLLM CUTLASS for NVFP4 (W4A4) and Humming
grouped experts for NVFP4A16 (W4A16). Automatic selection chooses these backends
when expert input transforms are present. An explicit incompatible backend raises
an error. NVFP4 keeps each expert's saved input global scale and applies the
vector before activation quantization. NVFP4A16 keeps activations in the model's
16-bit dtype. Gate and up projections must have matching vectors to share their
fused input. Ungated up/down experts are also represented by this loader.

The first implementation covers per-expert projection checkpoints, single-GPU
execution and tensor-parallel vector sharding. Expert parallelism, expert load
balancing, and fused shared-expert execution are not implemented. Shared experts
executed as ordinary linear layers use the dense transform path. Native CUTLASS
NVFP4 requires a GPU supported by that backend.

GPU validation is required before treating this development branch as verified.
On two supported NVIDIA GPUs, run:

```bash
.venv/bin/python -m pytest -v tests/quantization/test_compressed_tensors_moe_transforms.py tests/quantization/test_moe_diagonal_loading.py
```

The native tests load real packed checkpoints and compare compensated expert
vectors with identity vectors for both formats, TP=1/2, prefill/decode, and eager
and CUDA graph execution. CPU tests cover vector bytes, metadata, routing offsets,
missing weights, aliases, padding, and TP slicing. These tests are correctness
checks, not model accuracy benchmarks or a replacement for LLMC calibration and
GPU evaluation of an exported model.
