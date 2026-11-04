# GGUF Custom Operators for ONNX Runtime (`gguf_onnx_ops`)

## Overview

`gguf_onnx_ops` provides CUDA custom operators (registered under both `bolt` and `gguf` operator domains) that consume raw **GGUF v2/v3 tensor layouts** directly in GPU memory without offline dequantization, repacking, or scale splitting.

When paired with ONNX Runtime's zero-copy external initializer injection (`AddExternalInitializers` + `session.use_external_initializer_file_buffers_directly = "1"`) and ONNX Runtime GenAI's `gguf_initializers` configuration, an ONNX model graph (`model.onnx`) requires **zero `.onnx.data` external weight storage** and executes directly against an unmodified `.gguf` file.

---

## Supported GGUF Tensor Formats

| GGUF Type | ID | Elements / Block (`blck_size`) | Bytes / Block (`type_size`) | Binary Block Layout | Default ONNX View |
| --- | ---: | ---: | ---: | --- | --- |
| `F32` | `0` | `1` | `4` | IEEE 754 `float32` | `FLOAT [D_{n-1}, ..., D_0]` |
| `Q8_0` | `8` | `32` | `34` | `fp16 d` (2 B) + `int8 qs[32]` (32 B) | `UINT8 [..., (K/32)*34]` |
| `IQ2_XXS` | `16` | `256` | `66` | `fp16 d` (2 B) + 8 sub-blocks of `(uint8 grid[4], uint32 signs_and_scale)` (64 B) | `UINT8 [..., (K/256)*66]` |
| `NVFP4` | `40` | `64` | `36` | `uint8 d[4]` (`UE4M3` block-16 scales, 4 B) + 4 sub-blocks of `uint8 qs[8]` (32 B) | `UINT8 [..., (K/64)*36]` |

---

## Operator Reference

### 1. `Q8Embed`
Gathers token rows from a GGUF `Q8_0` embedding table (`token_embd.weight`) and dequantizes them to `BFLOAT16` on the fly.

- **Inputs**:
  - `w` (`UINT8 [vocab_size, (K / 32) * 34]`): Raw GGUF `Q8_0` embedding table.
  - `input_ids` (`INT64 [...]`): Token IDs of arbitrary rank (e.g. `[batch_size, seq_len]`). Out-of-range IDs clamp to row 0.
- **Outputs**:
  - `output` (`BFLOAT16 [..., K]`): Dequantized embeddings where `output[..., k] = bf16(fp32(d) * q[k])`.

### 2. `Q8Gemv` (`gguf = 1`)
Computes `y = x @ W^T` where `W` is stored in raw GGUF `Q8_0` blocks (`34` bytes per `32` elements).

- **Attributes**:
  - `gguf` (`int`, default `0`): Set to `1` for raw GGUF `Q8_0` layout (`[N, (K / 32) * 34]`).
- **Inputs**:
  - `x` (`BFLOAT16 [..., K]`): Input activations with `1 <= R <= 8` flattened rows and `K % 64 == 0`.
  - `w` (`UINT8 [N, (K / 32) * 34]`): Raw GGUF `Q8_0` weight matrix.
  - `scales` (`BFLOAT16`, optional): Omitted (`""`) when `gguf = 1`.
- **Outputs**:
  - `y` (`BFLOAT16 [..., N]`): Output activations.
- **Implementation**:
  - Uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` tensor cores across `R = 1..8` rows.
  - Because two consecutive 34-byte `Q8_0` blocks span 68 bytes (`17` `uint32_t` words at a 4-byte-aligned address when `K % 64 == 0`), each 4-thread quarter-warp cooperatively loads 17 aligned `uint32_t` words without divergent branches and extracts both scales and 64 `int8` weights using `__shfl_sync` and `__byte_perm`.

### 3. `Q4Gemv` (`iq2 = 3`)
Computes `y = x @ W^T` where `W` is stored in raw GGUF `IQ2_XXS` super-blocks (`66` bytes per `256` elements).

- **Attributes**:
  - `iq2` (`int`, default `0`): Set to `3` for raw GGUF `IQ2_XXS` super-blocks (`[N, (K / 256) * 66]`).
- **Inputs**:
  - `x` (`BFLOAT16 [..., K]`): Input activations with `1 <= R <= 8` rows and `K % 256 == 0`.
  - `w` (`UINT8 [N, (K / 256) * 66]`): Raw GGUF `IQ2_XXS` weight matrix.
  - `scales` (`BFLOAT16`, optional): Omitted (`""`) when `iq2 = 3`.
- **Outputs**:
  - `y` (`BFLOAT16 [..., N]`): Output activations.
- **Implementation**:
  - Each thread loads its 16-element slice (`2` codebook grid bytes + `14` sign bits) and packs them into a single 32-bit register across the double-buffered software pipeline (`StageT`), reducing live register pressure by 24 registers/thread and decoding via the pre-expanded `kIq2xxsGridBf16` codebook in `DequantRow`.

### 4. `Nvfp4Gemv` (`gguf = 1`)
Computes `y = x @ W^T` where `W` is stored in raw GGUF `NVFP4` blocks (`36` bytes per `64` elements) plus a scalar `F32` global scale (`weight_scale`).

- **Attributes**:
  - `gguf` (`int`, default `0`): Set to `1` for raw GGUF `NVFP4` layout (`[N, (K / 64) * 36]`).
- **Inputs**:
  - `x` (`BFLOAT16 [..., K]`): Input activations with `1 <= R <= 8` rows and `K % 64 == 0`.
  - `w` (`UINT8 [N, (K / 64) * 36]`): Raw GGUF `NVFP4` weight matrix.
  - `w_scale` (`UINT8`, optional): Omitted (`""`) when `gguf = 1`.
  - `gscale` (`FLOAT [1]`): Tensor-wide global scale (`blk.{l}.ssm_{in,out}.weight_scale`).
- **Outputs**:
  - `y` (`BFLOAT16 [..., N]`): Output activations.
- **Implementation**:
  - Uses `WIDE = true` (128-element K chunks per warp step) and de-interleaves GGUF's split low/high nibble ordering (`q[0..7]` low nibbles, `q[8..15]` high nibbles) in `DequantRow` via `__byte_perm` before `m16n8k16` tensor-core accumulation.

### 5. `MoeVerify` (`iq2 = 3`)
Computes routed Mixture-of-Experts (`relu2` activation) for `1 <= R <= 8` rows directly from raw 3D GGUF `IQ2_XXS` expert tensors (`blk.{l}.ffn_{up,down}_exps.weight`).

- **Attributes**:
  - `topk` (`int`): Number of selected experts per token (e.g. `9`).
  - `iq2` (`int`): Set to `3` for raw GGUF `IQ2_XXS` expert tensors.
- **Inputs**:
  - `z` (`BFLOAT16 [R, 1024]`): Latent input activations.
  - `idx` (`INT32 [R, topk]`): Selected expert indices per row.
  - `weights` (`FLOAT [R, topk]`): Normalized routing weights per row.
  - `fc1_w` (`UINT8 [E, I, (1024 / 256) * 66]`): Raw GGUF `ffn_up_exps.weight`.
  - `fc1_s` (`BFLOAT16`, optional): Omitted (`""`) when `iq2 = 3`.
  - `fc2_w` (`UINT8 [E, 1024, (I / 256) * 66]`): Raw GGUF `ffn_down_exps.weight` (`I % 256 == 0`).
  - `fc2_s` (`BFLOAT16`, optional): Omitted (`""`) when `iq2 = 3`.
- **Outputs**:
  - `out` (`BFLOAT16 [R, 1024]`): Aggregated expert outputs.
  - `h_scratch` (`BFLOAT16 [R * topk, I]`): Intermediate `relu2` activations.
  - `part_scratch` (`FLOAT [R * topk, 1024]`): Per-(row, expert) partial outputs.
  - `meta_scratch` (`INT32 [1 + R * topk * 9]`): Unique-expert grouping metadata.

### 6. `MoeRouteGGUF`
Computes DeepSeek-style sigmoid + bias-corrected top-k MoE routing directly from `F32` GGUF router weights (`ffn_gate_inp.weight`) and bias (`exp_probs_b.bias`).

- **Attributes**:
  - `topk` (`int`): Number of experts per token.
  - `scale` (`float`): Routed scaling factor.
- **Inputs**:
  - `x` (`BFLOAT16 [R, K]`): Hidden states (`K % 256 == 0`).
  - `gate_w` (`FLOAT [E, K]`): GGUF `F32` router weights (rounded to `bf16` in registers).
  - `bias` (`FLOAT [E]`): GGUF `F32` score correction bias.
- **Outputs**:
  - `idx` (`INT32 [R, topk]`), `weights` (`FLOAT [R, topk]`), `scores` (`FLOAT [R, E]`).
- **Optimization**:
  - For `R = 8`, launches `E` thread blocks of 4 warps splitting `K`, reading `gate_w[e, :]` once from HBM across all 8 rows (`3.87x` faster than per-row launches).

### 7. `MambaStepGGUF` and `MambaVerifyGGUF`
Fused Mamba2 mixer step (`R <= 16`) and speculative verification step (`R <= 8` with per-row state rollback buffer) accepting `F32` GGUF `ssm_conv1d.weight` and `ssm_conv1d.bias` directly.

### 8. `RmsNorm` and `SkipRmsNorm`
Fused RMSNormalization and Residual-Add + RMSNormalization accepting `BFLOAT16` activations and `FLOAT` (`F32`) GGUF gamma weights (`H % 8 == 0`, with a dedicated single-load fast path for `H = 4096`).
