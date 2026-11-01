# LLM Compressor

[LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/) is a library for optimizing models for deployment with vLLM.
It provides a comprehensive set of quantization algorithms, including support for techniques such as FP4, FP8, INT8, and INT4 quantization.

## Why use LLM Compressor?

Modern LLMs often contain billions of parameters stored in 16-bit or 32-bit floating point, requiring substantial GPU memory and limiting deployment options.
Quantization lowers memory requirements while maintaining inference output quality by reducing the precision of model weights and activations to smaller data types.

LLM Compressor provides the following benefits:

- **Reduced memory footprint**: Run larger models on smaller GPUs.
- **Lower inference costs**: Serve more concurrent users per GPU, directly reducing the cost per query in production deployments.
- **Faster inference**: Smaller data types mean less memory bandwidth consumed, which often translates to higher throughput, especially for memory-bound workloads.

LLM Compressor handles the complexity of quantization, calibration, and format conversion, producing models ready for immediate use with vLLM.

## Key features

- **Multiple Quantization Algorithms**: Support for AWQ, GPTQ, AutoRound, and Round-to-Nearest.
Also includes support for QuIP and SpinQuant-style transforms as well as KV cache and attention quantization.
- **Multiple Quantization Methods**: Support for FP8, INT8, INT4, NVFP4, MXFP4, and mixed-precision quantization
- **One-Shot Quantization**: Quantize models quickly with minimal calibration data
- **vLLM Integration**: Seamlessly deploy quantized models with vLLM using the compressed-tensors format
- **Hugging Face Compatibility**: Works with models from the Hugging Face Hub

## Diagonal input transforms

Compressed-tensors checkpoints can describe per-channel input scaling using
`type: "diagonal"` in `quantization_config.transform_config.config_groups`.
vLLM loads the scale from `<layer>.<group>_input.weight` and multiplies each
input channel before the underlying Linear method quantizes or consumes it.
The saved vector is an input multiplier (`inverse: false`); inverse transforms
are unsupported.
For example, OSFP4's `osfp4` group loads `<layer>.osfp4_input.weight`.
The checkpoint weights must already contain the inverse balancing; vLLM does
not apply that balancing again.

Use the usual `quantization="compressed-tensors"` option, or allow vLLM to
detect it from the checkpoint. No OSFP4 plugin is required. The transform
retains its declared precision when loaded and converts the scale to the input
dtype for multiplication. Existing NVFP4 weight and activation scales retain
their usual meaning.

The initial implementation supports full-width input transforms on unfused
Linear projections. The vector follows the input-channel partition for row
parallelism and is replicated for column parallelism. Fused projections such
as `qkv_proj` and `gate_up_proj`, output transforms, and `head_dim` are not
supported. Existing quantization backend and hardware requirements still apply.

## Resources

- [LLM Compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples)
- [GitHub Repository](https://github.com/vllm-project/llm-compressor)
