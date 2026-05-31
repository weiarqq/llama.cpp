# SYCL XMX Quantized Matmul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime-gated XMX MMQ dispatch path for quantized SYCL `MUL_MAT` and `MUL_MAT_ID` while preserving fallback on non-XMX hardware.

**Architecture:** Keep the change localized to SYCL device feature tracking and `ggml_sycl_mul_mat()` dispatch. Reuse the existing `ggml_sycl_op_mul_mat_q` kernel path behind a stricter XMX capability gate and env toggle.

**Tech Stack:** C++17, SYCL/oneAPI, existing llama.cpp `test-backend-ops` backend validation.

---

## File Structure

- Modify `ggml/src/ggml-sycl/common.hpp`: add `optimize_feature::xmx`, declare `g_ggml_sycl_disable_xmx_mmq`.
- Modify `ggml/src/ggml-sycl/ggml-sycl.cpp`: initialize XMX feature, print it, read env var, add type/shape helper, gate MMQ dispatch.

## Task 1: Runtime XMX Feature Tracking

- [ ] Add `xmx` to `optimize_feature` in `common.hpp`.
- [ ] Set `info.devices[i].opt_feature.xmx = gpu_has_xmx(device)` during SYCL init.
- [ ] Print `XMX` in the optimization feature table.
- [ ] Build `ggml/src/ggml-sycl` to catch compile errors.

## Task 2: XMX MMQ Dispatch Gate

- [ ] Add `g_ggml_sycl_disable_xmx_mmq` and read `GGML_SYCL_DISABLE_XMX_MMQ` in `ggml_check_sycl()`.
- [ ] Add `ggml_sycl_supports_xmx_mmq_type()` for `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`.
- [ ] Add `ggml_sycl_can_use_xmx_mmq()` to require XMX feature, supported type, `F32` activation/output, non-split execution, and `src1->ne[1] <= MMQ_MAX_BATCH_SIZE`.
- [ ] Use this helper to enable `use_mul_mat_q` in `ggml_sycl_mul_mat()`.

## Task 3: Verification

- [ ] Configure a SYCL build with `cmake -B build-sycl -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=OFF`.
- [ ] Build `test-backend-ops` with `cmake --build build-sycl --target test-backend-ops -j`.
- [ ] Run correctness with `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -j 1`.
- [ ] Run correctness with `ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT_ID -j 1`.
- [ ] Run fallback comparison with `GGML_SYCL_DISABLE_XMX_MMQ=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -j 1`.
