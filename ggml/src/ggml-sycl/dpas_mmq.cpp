//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "dpas_mmq.hpp"

#include <sycl/ext/oneapi/experimental/matrix.hpp>

namespace matrix = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

// ---------------------------------------------------------------------------
// Q4_0: unpack 32 nibbles → bf16 × scale
// Each Q4_0 block: 1 f16 scale + 16 bytes of nibbles (32 × 4-bit)
// Unpack to bf16 in-place into a local buffer.
// ---------------------------------------------------------------------------
static inline void q4_0_to_bf16(
    const block_q4_0 &block,
    bf16 out[QK4_0],
    bf16 scale_out)
{
    float d_f32 = __half2float(block.d);
    scale_out = bf16(d_f32);

    for (int i = 0; i < 16; i++) {
        uint8_t byte = block.qs[i];
        // Q4_0: unsigned 4-bit, dequantize formula = (value - 8) * d
        // We keep as (value - 8) in bf16 and multiply by scale later
        out[2 * i + 0] = bf16(static_cast<float>(static_cast<int8_t>((byte & 0x0F) - 8)));
        out[2 * i + 1] = bf16(static_cast<float>(static_cast<int8_t>((byte >> 4) - 8)));
    }
}

// ---------------------------------------------------------------------------
// Q8_1: load int8 quant → bf16 × scale
// Q8_1 block: 1 f16 scale + 1 f16 sum + 32 int8 values
// ---------------------------------------------------------------------------
static inline void q8_1_to_bf16(
    const block_q8_1 &block,
    bf16 out[QK8_1],
    bf16 scale_out)
{
    float d_f32 = __half2float(block.d);
    scale_out = bf16(d_f32);

    for (int i = 0; i < QK8_1; i++) {
        out[i] = bf16(static_cast<float>(block.qs[i]));
    }
}

// ---------------------------------------------------------------------------
// DPAS kernel: Q4_0 × Q8_1 via joint_matrix (bf16 × bf16 → f32)
//
// Each work-group processes a DPAS_WG_TILE_M × DPAS_WG_TILE_N tile of C.
// Each sub_group handles DPAS_TILE_M × DPAS_TILE_N.
// K loop steps by QK4_0 (=32), with joint_matrix inner dimension = DPAS_TILE_K.
// ---------------------------------------------------------------------------
template <bool need_check>
static void dpas_mul_mat_q4_0_q8_1(
    const block_q4_0 *__restrict__ src0,
    const block_q8_1 *__restrict__ src1,
    float *__restrict__ dst,
    int M, int N, int K,
    int blocks_per_row)
{
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item.get_sub_group();

    // Work-group base
    const int wg_m = item.get_group(2) * DPAS_WG_TILE_M;
    const int wg_n = item.get_group(1) * DPAS_WG_TILE_N;

    // Sub-group indices
    const int sg_row = item.get_local_id(1); // 0..DPAS_WG_SG_M-1
    const int sg_col = 0; // only 1 sub_group in N dim

    const int mb = wg_m + sg_row * DPAS_TILE_M;
    const int nb = wg_n + sg_col * DPAS_TILE_N;

    // joint_matrix objects
    matrix::joint_matrix<bf16, matrix::use::a, DPAS_TILE_M, DPAS_TILE_N, DPAS_TILE_K,
        matrix::layout::row_major> tma;
    matrix::joint_matrix<bf16, matrix::use::b, DPAS_TILE_N, DPAS_TILE_M, DPAS_TILE_K,
        matrix::layout::col_major> tmb;
    matrix::joint_matrix<float, matrix::use::accumulator, DPAS_TILE_M, DPAS_TILE_N> tmc;

    matrix::joint_matrix_fill(sg, tmc, 0.0f);

    // Local buffers for tile data
    bf16 a_buf[DPAS_TILE_M * QK4_0];
    bf16 b_buf[DPAS_TILE_N * QK4_0];

    // K loop: 1 iteration = 1 Q4_0 block = 32 K elements
    // joint_matrix processes 16 K elements per mad, so 2 mads per block
    constexpr int K_PER_BLOCK = QK4_0;
    constexpr int JM_ITER = K_PER_BLOCK / DPAS_TILE_K; // 2

    for (int kb = 0; kb < K_PER_BLOCK * blocks_per_row; kb += K_PER_BLOCK) {
        int block_id = kb / K_PER_BLOCK;

        // --- Load A tile (Q4_0 → bf16) ---
        for (int r = 0; r < DPAS_TILE_M; r++) {
            int row = mb + r;
            if constexpr (need_check) {
                if (row >= M) break;
            }
            const block_q4_0 &blk = src0[row * blocks_per_row + block_id];
            bf16 scale;
            bf16 unpacked[QK4_0];
            q4_0_to_bf16(blk, unpacked, scale);
            for (int kk = 0; kk < QK4_0; kk++) {
                a_buf[r * QK4_0 + kk] = unpacked[kk] * scale;
            }
        }

        // --- Load B tile (Q8_1 → bf16) ---
        for (int c = 0; c < DPAS_TILE_N; c++) {
            int col = nb + c;
            if constexpr (need_check) {
                if (col >= N) break;
            }
            const block_q8_1 &blk = src1[col * blocks_per_row + block_id];
            bf16 scale;
            bf16 unpacked[QK8_1];
            q8_1_to_bf16(blk, unpacked, scale);
            for (int kk = 0; kk < QK8_1; kk++) {
                // col-major layout: B[kk][c] = kk * N + c
                b_buf[kk * DPAS_TILE_N + c] = unpacked[kk] * scale;
            }
        }

        // joint_matrix compute: 2 iterations (K=32, JM_K=16)
        for (int jm = 0; jm < JM_ITER; jm++) {
            int k_off = jm * DPAS_TILE_K;
            matrix::joint_matrix_load(sg, tma,
                a_buf + k_off,              // A tile starts at this K offset
                QK4_0,                      // stride = full K per block
                matrix::layout::row_major);
            matrix::joint_matrix_load(sg, tmb,
                b_buf + k_off * DPAS_TILE_N, // B[k_off][0] in col-major
                DPAS_TILE_N,                // stride = N
                matrix::layout::col_major);
            matrix::joint_matrix_mad(sg, tmc, tma, tmb, tmc);
        }
    }

    // --- Store result, with DPAS_TILE_M × DPAS_TILE_N f32 output ---
    float c_buf[DPAS_TILE_M * DPAS_TILE_N];
    matrix::joint_matrix_store(sg, tmc, c_buf,
        DPAS_TILE_N, matrix::layout::row_major);

    for (int r = 0; r < DPAS_TILE_M; r++) {
        int row = mb + r;
        if constexpr (need_check) {
            if (row >= M) break;
        }
        for (int c = 0; c < DPAS_TILE_N; c++) {
            int col = nb + c;
            if constexpr (need_check) {
                if (col >= N) break;
            }
            dst[row * N + col] = c_buf[r * DPAS_TILE_N + c];
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0 × Q8_1 via joint_matrix (simpler: no nibble unpack)
// ---------------------------------------------------------------------------
template <bool need_check>
static void dpas_mul_mat_q8_0_q8_1(
    const block_q8_0 *__restrict__ src0,
    const block_q8_1 *__restrict__ src1,
    float *__restrict__ dst,
    int M, int N, int K,
    int blocks_per_row)
{
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item.get_sub_group();

    const int wg_m = item.get_group(2) * DPAS_WG_TILE_M;
    const int wg_n = item.get_group(1) * DPAS_WG_TILE_N;
    const int sg_row = item.get_local_id(1);
    const int mb = wg_m + sg_row * DPAS_TILE_M;
    const int nb = wg_n;

    matrix::joint_matrix<bf16, matrix::use::a, DPAS_TILE_M, DPAS_TILE_N, DPAS_TILE_K,
        matrix::layout::row_major> tma;
    matrix::joint_matrix<bf16, matrix::use::b, DPAS_TILE_N, DPAS_TILE_M, DPAS_TILE_K,
        matrix::layout::col_major> tmb;
    matrix::joint_matrix<float, matrix::use::accumulator, DPAS_TILE_M, DPAS_TILE_N> tmc;

    matrix::joint_matrix_fill(sg, tmc, 0.0f);

    bf16 a_buf[DPAS_TILE_M * QK8_0];
    bf16 b_buf[DPAS_TILE_N * QK8_0];
    constexpr int JM_ITER = QK8_0 / DPAS_TILE_K;

    for (int kb = 0; kb < QK8_0 * blocks_per_row; kb += QK8_0) {
        int block_id = kb / QK8_0;

        // A tile: Q8_0 → bf16 (each Q8_0 has f16 scale + 32 int8)
        for (int r = 0; r < DPAS_TILE_M; r++) {
            int row = mb + r;
            if constexpr (need_check) {
                if (row >= M) break;
            }
            const block_q8_0 &blk = src0[row * blocks_per_row + block_id];
            float d = __half2float(blk.d);
            bf16 scale = bf16(d);
            for (int kk = 0; kk < QK8_0; kk++) {
                a_buf[r * QK8_0 + kk] = bf16(static_cast<float>(blk.qs[kk])) * scale;
            }
        }

        // B tile: Q8_1 → bf16
        for (int c = 0; c < DPAS_TILE_N; c++) {
            int col = nb + c;
            if constexpr (need_check) {
                if (col >= N) break;
            }
            const block_q8_1 &blk = src1[col * blocks_per_row + block_id];
            float d = __half2float(blk.d);
            bf16 scale = bf16(d);
            for (int kk = 0; kk < QK8_1; kk++) {
                b_buf[kk * DPAS_TILE_N + c] = bf16(static_cast<float>(blk.qs[kk])) * scale;
            }
        }

        for (int jm = 0; jm < JM_ITER; jm++) {
            int k_off = jm * DPAS_TILE_K;
            matrix::joint_matrix_load(sg, tma,
                a_buf + k_off, QK8_0, matrix::layout::row_major);
            matrix::joint_matrix_load(sg, tmb,
                b_buf + k_off * DPAS_TILE_N, DPAS_TILE_N, matrix::layout::col_major);
            matrix::joint_matrix_mad(sg, tmc, tma, tmb, tmc);
        }
    }

    float c_buf[DPAS_TILE_M * DPAS_TILE_N];
    matrix::joint_matrix_store(sg, tmc, c_buf,
        DPAS_TILE_N, matrix::layout::row_major);

    for (int r = 0; r < DPAS_TILE_M; r++) {
        int row = mb + r;
        if constexpr (need_check) {
            if (row >= M) break;
        }
        for (int c = 0; c < DPAS_TILE_N; c++) {
            int col = nb + c;
            if constexpr (need_check) {
                if (col >= N) break;
            }
            dst[row * N + col] = c_buf[r * DPAS_TILE_N + c];
        }
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------
bool gpu_has_dpas(sycl::device &dev) {
    return dev.has(sycl::aspect::ext_intel_matrix);
}

// ---------------------------------------------------------------------------
// Dispatch entry point
// ---------------------------------------------------------------------------
void ggml_sycl_op_mul_mat_dpas(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream) try
{
    const int64_t ne00 = src0->ne[0];
    GGML_ASSERT(ne00 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    int device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));
    const int64_t nrows_dst = (device_id == ctx.device) ? ne0 : row_diff;

    const int M = static_cast<int>(row_diff);
    const int N = static_cast<int>(src1_ncols);
    const int K = static_cast<int>(ne00);
    const int blocks_per_row = K / QK4_0;

    // Truncate N to tile boundary for simplicity; remainder handled by need_check.
    const int grid_m = (M + DPAS_WG_TILE_M - 1) / DPAS_WG_TILE_M;
    const int grid_n = (N + DPAS_WG_TILE_N - 1) / DPAS_WG_TILE_N;

    sycl::range<3> local(1, DPAS_WG_SG_M,
                         DPAS_WG_SG_N * WARP_SIZE); // WG_SG_N=1, so =16
    sycl::range<3> global(1, grid_m * DPAS_WG_SG_M,
                          grid_n * DPAS_WG_SG_N * WARP_SIZE);

    auto q = *stream;
    q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<3>(global, local),
            [=](sycl::nd_item<3> it) {
                const bool need_check
                    = (M < grid_m * DPAS_WG_TILE_M)
                   || (N < grid_n * DPAS_WG_TILE_N);

                switch (src0->type) {
                    case GGML_TYPE_Q4_0: {
                        auto A = reinterpret_cast<const block_q4_0 *>(src0_dd_i);
                        auto B = reinterpret_cast<const block_q8_1 *>(src1_ddq_i);
                        auto C = dst_dd_i;
                        if (need_check) {
                            dpas_mul_mat_q4_0_q8_1<true>(
                                A, B, C, M, N, K, blocks_per_row);
                        } else {
                            dpas_mul_mat_q4_0_q8_1<false>(
                                A, B, C, M, N, K, blocks_per_row);
                        }
                        break;
                    }
                    case GGML_TYPE_Q8_0: {
                        auto A = reinterpret_cast<const block_q8_0 *>(src0_dd_i);
                        auto B = reinterpret_cast<const block_q8_1 *>(src1_ddq_i);
                        auto C = dst_dd_i;
                        if (need_check) {
                            dpas_mul_mat_q8_0_q8_1<true>(
                                A, B, C, M, N, K, blocks_per_row);
                        } else {
                            dpas_mul_mat_q8_0_q8_1<false>(
                                A, B, C, M, N, K, blocks_per_row);
                        }
                        break;
                    }
                    default:
                        // Not yet implemented for other quant types
                        break;
                }
            });
    });

    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(nrows_dst);
}
catch (sycl::exception const &exc) {
    std::cerr << exc.what() << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    std::exit(1);
}
