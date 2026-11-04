# gguf_onnx_ops

CUDA custom-operator library for running ONNX models directly on **native GGUF weight tensors** (`IQ2_XXS`, `NVFP4`, `Q8_0`, `F32`) with ONNX Runtime's CUDA Execution Provider and ONNX Runtime GenAI.

Built on the ONNX Runtime C/C++ custom-op ABI (`sm_90`), all kernels execute on the provider's compute stream, are **CUDA-Graph capturable**, and register under both the `bolt` and `gguf` custom operator domains.

See [docs/GGUF_ONNX_Custom_Ops.md](./docs/GGUF_ONNX_Custom_Ops.md) for detailed operator specifications, GGUF block binary layouts, and `gguf_initializers.json` integration.

## Build

```bash
export PATH=$CUDA_HOME/bin:$PATH
cmake -S . -B build \
  -DORT_INCLUDE_DIR=<ort>/include/onnxruntime/core/session \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build build -j
# Produces build/lib/libbolt_ops.so and build/lib/libgguf_onnx_ops.so
```

## Test

```bash
BOLT_PLUGIN=<libonnxruntime_providers_cuda.so> \
BOLT_OPS=build/lib/libbolt_ops.so \
python tests/test_bolt_ops.py
```

## Operators Summary (`bolt` and `gguf` domains)

| Operator | GGUF Weight Format | Description | Inputs -> Outputs |
| --- | --- | --- | --- |
| `Q8Embed` | `Q8_0` (`34 B / 32 elems`) | Gather + dequantize embedding rows directly from GGUF `token_embd.weight` | `w [V, (K/32)*34]` u8, `ids [...]` i64 -> `out [..., K]` bf16 |
| `Q8Gemv` (`gguf=1`) | `Q8_0` (`34 B / 32 elems`) | Tensor-core (`mma.m16n8k16`) & warp GEMV on raw 34-byte GGUF `Q8_0` blocks (no separate scale tensor) | `x [..., K]` bf16, `w [N, (K/32)*34]` u8 -> `y [..., N]` bf16 |
| `Q4Gemv` (`iq2=3`) | `IQ2_XXS` (`66 B / 256 elems`) | Tensor-core GEMV on raw 66-byte GGUF `IQ2_XXS` super-blocks (no separate scale tensor) | `x [..., K]` bf16, `w [N, (K/256)*66]` u8 -> `y [..., N]` bf16 |
| `Nvfp4Gemv` (`gguf=1`) | `NVFP4` (`36 B / 64 elems`) | Tensor-core GEMV on raw 36-byte GGUF `NVFP4` blocks + `F32` global scale | `x [..., K]` bf16, `w [N, (K/64)*36]` u8, `""`, `gscale [1]` f32 -> `y [..., N]` bf16 |
| `MoeVerify` (`iq2=3`) | `IQ2_XXS` (`66 B / 256 elems`) | Expert-grouped tensor-core MoE (`relu2`) on raw 66-byte GGUF `IQ2_XXS` up/down expert tensors (`R <= 8`) | `z [R,1024]`, `idx [R,k]`, `w [R,k]`, `w1 [E,I,(1024/256)*66]`, `""`, `w2 [E,1024,(I/256)*66]` -> `out [R,1024]` (+3 scratch) |
| `MoeRouteGGUF` | `F32` | MoE router with `F32` GGUF gate weights (`ffn_gate_inp.weight`) and bias (`exp_probs_b.bias`) | `x [R,K]` bf16, `gate_w [E,K]` f32, `bias [E]` f32 -> `idx [R,k]` i32, `w [R,k]` f32, `scores [R,E]` f32 |
| `MambaStepGGUF` | `F32` | Fused Mamba2 mixer step (`R <= 16`) with `F32` GGUF `conv1d_w`, `conv1d_b`, `dt_bias`, `A_neg`, `D`, `norm_w` | `proj`, `conv_state`, `ssm_state`, `conv_w`, `conv_b`, `dt_bias`, `A_neg`, `D`, `norm_w` -> `y`, `conv_state'`, `ssm_state'` (+2 scratch) |
| `MambaVerifyGGUF` | `F32` | Speculative verification `MambaStepGGUF` with per-row `[Rs, ...]` state buffers and device `state_row` selector | as `MambaStepGGUF` + `state_row [1]` i32 |
| `RmsNorm` | `F32` | RMSNormalization with `BF16` activations and `F32` GGUF gamma weights (`H % 8 == 0`) | `x [..., H]` bf16, `gamma [H]` f32 -> `y [..., H]` bf16 |
| `SkipRmsNorm` | `F32` | Fused residual-add + RMSNormalization with `BF16` activations and `F32` GGUF gamma weights | `x [..., H]` bf16, `skip [..., H]` bf16, `gamma [H]` f32 -> `y [..., H]` bf16, optional `sum_out [..., H]` bf16 |

In addition, the library preserves all legacy non-GGUF operators (`MambaStep`, `MambaVerify`, `MoeRoute`, `MoeDecode`, `MoeVerify` with `iq2=0/1/2`, `Nvfp4Gemv` with `gguf=0`, `Q8Gemv` with `gguf=0`, `Q4Gemv` with `iq2=0/1/2`, and `Relu2`) for full backward compatibility.
