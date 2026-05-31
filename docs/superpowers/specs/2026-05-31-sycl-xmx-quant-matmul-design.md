# SYCL XMX Quantized Matmul Design

## Goal

Add a runtime-gated XMX path for SYCL `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` when running quantized weights with `F32` activations on Intel Arc/Xe HPG class GPUs, while preserving existing behavior on devices without XMX.

## Scope

The first implementation targets quantized `src0` types `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K` with `src1` and `dst` as `F32`. `MUL_MAT_ID` should reuse the existing expert grouping path because it already calls `ggml_sycl_mul_mat()` for each expert slice.

## Dispatch Design

The dispatch hook lives in `ggml_sycl_mul_mat()` near the existing MMQ decision. A new helper decides whether XMX MMQ is allowed for the current device, type, and shape. The helper must use runtime hardware capability, not only the compile-time `SYCL_USE_XMX` macro.

The path is enabled only when all of these are true:

- The selected device reports `sycl::aspect::ext_intel_matrix` through `gpu_has_xmx()`.
- The weight type is one of `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, or `Q6_K`.
- `src1->type == GGML_TYPE_F32` and `dst->type == GGML_TYPE_F32`.
- `src1->ne[1] <= MMQ_MAX_BATCH_SIZE`.
- The operation is not a split-buffer multi-GPU operation.
- The user has not disabled the path with `GGML_SYCL_DISABLE_XMX_MMQ=1`.

If any condition fails, dispatch falls through to existing DMMV/MMVQ/reorder/generic SYCL paths.

## Compatibility

Non-XMX devices must keep the previous behavior. The existing `SYCL_USE_XMX` macro remains a compile-time guard, but runtime gating is mandatory. The new env toggle allows correctness and performance A/B testing without rebuilding.

## Testing

Correctness should be checked with `test-backend-ops` filters for `MUL_MAT` and `MUL_MAT_ID` on SYCL. Performance should be measured with `perf` mode before and after enabling `GGML_SYCL_DISABLE_XMX_MMQ=0`. Fallback should be verified by running the same tests with `GGML_SYCL_DISABLE_XMX_MMQ=1` or on a non-XMX device.
