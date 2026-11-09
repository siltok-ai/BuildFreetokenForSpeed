# Attribution and modification notices

## Model weights and tokenizer

These GGUF files are converted and quantized versions of [DiffuSpace/Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), a text-only block diffusion model derived from [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

Upstream Qwen model assets are Copyright 2026 Alibaba Cloud and distributed under Apache-2.0. The accompanying [LICENSE](./LICENSE) preserves the upstream license and copyright statement. Clover adds learned parameters for bidirectional Gated DeltaNet computation and block diffusion generation. See the [original model's attribution notices](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/ATTRIBUTIONS.md) for source implementation provenance.

## GGUF conversion and quantization

The models are packaged for the Clover integration in [llama.cpp PR #31306](https://github.com/ggml-org/llama.cpp/pull/31306). The three variants use Q8_0, mixed Q4_K_M, and Q4_0 quantization respectively; not every tensor has the same precision.

Release metadata uses the model name `Clover-35B-A3B-Preview`, the release version `Preview`, and a maximum context length of 131,072 tokens. Correcting these metadata fields does not modify the quantized tensor data.

llama.cpp is a separate project distributed under the [MIT license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE). Its software license does not replace the Apache-2.0 license of these model assets.
