//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Intel XMX/DPAS quantized matmul (Q4_0, Q8_0, etc.)
//
// Design:
//   Each sub_group processes a 16×32 tile of C via joint_matrix (bf16 × bf16 → f32).
//   Q4_0 nibbles are unpacked to bf16 in registers before loading into joint_matrix.
//   Scale is applied per Q4_0 block after joint_matrix accumulation.
//
// Requirements:
//   Intel GPU with XMX: sycl::device::has(sycl::aspect::ext_intel_matrix)
//   Intel oneAPI 2024+ with joint_matrix support
//

#ifndef GGML_SYCL_DPAS_MMQ_HPP
#define GGML_SYCL_DPAS_MMQ_HPP

#include "common.hpp"

// ---------------------------------------------------------------------------
// Tile configuration
// ---------------------------------------------------------------------------
// joint_matrix tile sizes for Intel XeHPC+ (bf16 × bf16 → f32):
//   A: 16×32 bf16  (sub-tile K=16, needs 2 iterations for Q4_0 block of 32)
//   B: 32×32 bf16
//   C: 16×32 f32
//
// Fallback for older Intel (XeHP):
//   A: 8×16 bf16
//   B: 16×16 bf16
//   C: 8×16 f32

// Primary tile
static constexpr int DPAS_TILE_M = 16;
static constexpr int DPAS_TILE_N = 32;
static constexpr int DPAS_TILE_K = 16;  // joint_matrix inner dim

// Work-group tiles (sub-groups in a WG)
static constexpr int DPAS_WG_SG_M = 2;   // 2 sub-groups in M → WG tile M = 32
static constexpr int DPAS_WG_SG_N = 1;   // 1 sub-group in N → WG tile N = 32

static constexpr int DPAS_WG_TILE_M = DPAS_TILE_M * DPAS_WG_SG_M;
static constexpr int DPAS_WG_TILE_N = DPAS_TILE_N * DPAS_WG_SG_N;

static constexpr int Q4_0_UNROLL = QK4_0; // 32, each Q4_0 block

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------
bool gpu_has_dpas(sycl::device &dev);

// ---------------------------------------------------------------------------
// Entry point — same signature as ggml_sycl_op_mul_mat_q
// ---------------------------------------------------------------------------
void ggml_sycl_op_mul_mat_dpas(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

#endif // GGML_SYCL_DPAS_MMQ_HPP
