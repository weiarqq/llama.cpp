#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

extern "C" {
struct ggml_moe_ffn_qwen35_decode_timings {
    int64_t cur_q_us;
    int64_t gate_up_us;
    int64_t barrier_us;
    int64_t act_q_us;
    int64_t down_us;
    int64_t calls;
};

void ggml_moe_ffn_qwen35_decode_timings_reset(void);
ggml_moe_ffn_qwen35_decode_timings ggml_moe_ffn_qwen35_decode_timings_get(void);
}

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        ggml_free(ctx);
    }
};

struct ggml_backend_deleter {
    void operator()(ggml_backend * backend) const {
        ggml_backend_free(backend);
    }
};

struct ggml_backend_buffer_deleter {
    void operator()(ggml_backend_buffer * buffer) const {
        ggml_backend_buffer_free(buffer);
    }
};

using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;
using ggml_backend_ptr = std::unique_ptr<ggml_backend, ggml_backend_deleter>;
using ggml_backend_buffer_ptr = std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter>;

static constexpr int64_t n_embd        = 2048;
static constexpr int64_t n_ff          = 512;
static constexpr int64_t n_embd_out    = 2048;
static constexpr int64_t n_expert      = 256;
static constexpr int64_t n_expert_used = 8;
static constexpr int     n_threads     = 16;

static float weight_value(int64_t i0, int64_t i1, int64_t i2, float phase) {
    const int64_t v = (i0*13 + i1*17 + i2*19 + static_cast<int64_t>(phase*100.0f)) % 37;
    return (static_cast<float>(v) - 18.0f)*0.015f;
}

static std::vector<float> make_f32_3d(int64_t ne0, int64_t ne1, int64_t ne2, float phase) {
    std::vector<float> data(ne0*ne1*ne2);
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                data[i0 + ne0*(i1 + ne1*i2)] = weight_value(i0, i1, i2, phase);
            }
        }
    }
    return data;
}

static std::vector<float> make_up_gate_f32_3d(const std::vector<float> & up, const std::vector<float> & gate) {
    std::vector<float> data(2*n_embd*n_ff*n_expert);
    for (int64_t i2 = 0; i2 < n_expert; ++i2) {
        for (int64_t i1 = 0; i1 < n_ff; ++i1) {
            const size_t src_off = n_embd*(i1 + n_ff*i2);
            const size_t dst_off = 2*n_embd*(i1 + n_ff*i2);
            memcpy(data.data() + dst_off,          up.data()   + src_off, n_embd*sizeof(float));
            memcpy(data.data() + dst_off + n_embd, gate.data() + src_off, n_embd*sizeof(float));
        }
    }
    return data;
}

static std::vector<uint8_t> quantize_3d(enum ggml_type type, const std::vector<float> & src, int64_t ne0, int64_t ne1, int64_t ne2) {
    std::vector<uint8_t> dst(ggml_row_size(type, ne0)*ne1*ne2);
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        const float * src_plane = src.data() + ne0*ne1*i2;
        uint8_t * dst_plane = dst.data() + ggml_row_size(type, ne0)*ne1*i2;
        ggml_quantize_chunk(type, src_plane, dst_plane, 0, ne1, ne0, nullptr);
    }
    return dst;
}

static void set_quantized_tensor(ggml_tensor * tensor, const std::vector<uint8_t> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
}

static ggml_tensor * new_weight_tensor(ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_name(tensor, "moe_weight");
    return tensor;
}


static size_t buffer_size_for_tensors(ggml_backend_buffer_type_t buft, ggml_context * ctx, ggml_tensor * skip0, ggml_tensor * skip1) {
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (tensor != skip0 && tensor != skip1 && tensor->data == nullptr && tensor->view_src == nullptr) {
            size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
        }
    }
    return size + alignment;
}

static ggml_context_ptr make_context() {
    const size_t graph_nodes = 64;
    ggml_init_params params = {
        /* .mem_size = */ 32*1024*1024 + ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(graph_nodes, false),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        fprintf(stderr, "failed to create ggml context\n");
        std::exit(1);
    }
    return ctx;
}

enum class moe_path {
    unfused,
    fused,
    gate_up_fused
};

struct run_result {
    std::vector<float> output;
    double             time_us;
    int                n_runs;
    ggml_moe_ffn_qwen35_decode_timings timings;
};

struct moe_graph {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr host_buffer;
    ggml_backend_ptr        backend;
    ggml_cgraph *           graph = nullptr;
    ggml_tensor *           out   = nullptr;
    moe_path                path  = moe_path::unfused;

    run_result run() {
        if (path == moe_path::fused || path == moe_path::gate_up_fused) {
            ggml_moe_ffn_qwen35_decode_timings_reset();
        }

        ggml_status status = ggml_backend_graph_compute(backend.get(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
            std::exit(1);
        }

        int64_t total_time_us = 0;
        int total_runs = 0;
        do {
            const int64_t start_time = ggml_time_us();
            status = ggml_backend_graph_compute(backend.get(), graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
                std::exit(1);
            }
            const int64_t end_time = ggml_time_us();
            total_time_us += end_time - start_time;
            ++total_runs;
        } while (total_time_us < 1000*1000);

        std::vector<float> output(ggml_nelements(out));
        ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(float));
        ggml_moe_ffn_qwen35_decode_timings timings = {};
        if (path == moe_path::fused || path == moe_path::gate_up_fused) {
            timings = ggml_moe_ffn_qwen35_decode_timings_get();
        }
        return { output, static_cast<double>(total_time_us)/total_runs, total_runs, timings };
    }
};

static moe_graph build_moe_graph(
        enum ggml_type down_type,
        moe_path       path,
        const std::vector<uint8_t> & up_q,
        const std::vector<uint8_t> & gate_q,
        const std::vector<uint8_t> & up_gate_q,
        const std::vector<uint8_t> & down_q,
        const std::vector<float>   & cur_data,
        const std::vector<int32_t> & ids_data) {
    moe_graph result;
    result.ctx = make_context();
    result.path = path;
    if (path != moe_path::gate_up_fused) {

        ggml_tensor * up_exps   = new_weight_tensor(result.ctx.get(), GGML_TYPE_Q4_K, n_embd, n_ff, n_expert);
        ggml_tensor * gate_exps = new_weight_tensor(result.ctx.get(), GGML_TYPE_Q4_K, n_embd, n_ff, n_expert);
        ggml_tensor * down_exps = new_weight_tensor(result.ctx.get(), down_type, n_ff, n_embd_out, n_expert);
        ggml_tensor * cur       = ggml_new_tensor_3d(result.ctx.get(), GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor * ids       = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_I32, n_expert_used, 1);

        if (path == moe_path::fused) {
            result.out = ggml_moe_ffn_qwen35_decode(result.ctx.get(), up_exps, gate_exps, down_exps, cur, ids);
        } else {
            ggml_tensor * up   = ggml_mul_mat_id(result.ctx.get(), up_exps, cur, ids);
            ggml_tensor * gate = ggml_mul_mat_id(result.ctx.get(), gate_exps, cur, ids);
            ggml_tensor * act  = ggml_swiglu_split(result.ctx.get(), gate, up);
            result.out = ggml_mul_mat_id(result.ctx.get(), down_exps, act, ids);
        }
        ggml_set_output(result.out);

        const bool repack = path == moe_path::fused;
        ggml_backend_buffer_type_t weight_buft = repack ? ggml_backend_cpu_repack_buffer_type() : ggml_backend_cpu_buffer_type();
        result.weight_buffer.reset(ggml_backend_buft_alloc_buffer(weight_buft,
                ggml_nbytes(up_exps) + ggml_nbytes(gate_exps) + 3*ggml_backend_buft_get_alignment(weight_buft)));
        ggml_backend_buffer_type_t host_buft = ggml_backend_cpu_buffer_type();
        result.host_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, buffer_size_for_tensors(host_buft, result.ctx.get(), up_exps, gate_exps)));
        if (!result.weight_buffer || !result.host_buffer) {
            fprintf(stderr, "failed to allocate tensor buffers\n");
            std::exit(1);
        }

        ggml_tallocr weight_alloc = ggml_tallocr_new(result.weight_buffer.get());
        ggml_tallocr host_alloc   = ggml_tallocr_new(result.host_buffer.get());
        ggml_tallocr_alloc(&weight_alloc, up_exps);
        ggml_tallocr_alloc(&weight_alloc, gate_exps);

        for (ggml_tensor * tensor = ggml_get_first_tensor(result.ctx.get()); tensor != nullptr; tensor = ggml_get_next_tensor(result.ctx.get(), tensor)) {
            if (tensor != up_exps && tensor != gate_exps && tensor->data == nullptr && tensor->view_src == nullptr) {
                ggml_tallocr_alloc(&host_alloc, tensor);
            }
        }

        set_quantized_tensor(up_exps, up_q);
        set_quantized_tensor(gate_exps, gate_q);
        set_quantized_tensor(down_exps, down_q);
        ggml_backend_tensor_set(cur, cur_data.data(), 0, cur_data.size()*sizeof(float));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));

        result.graph = ggml_new_graph_custom(result.ctx.get(), 64, false);
        ggml_build_forward_expand(result.graph, result.out);

        result.backend.reset(ggml_backend_cpu_init());
        ggml_backend_cpu_set_n_threads(result.backend.get(), n_threads);
        return result;
    } else {

        ggml_tensor * up_gate_exps = new_weight_tensor(result.ctx.get(), GGML_TYPE_Q4_K, 2*n_embd, n_ff, n_expert);
        ggml_tensor * down_exps = new_weight_tensor(result.ctx.get(), down_type, n_ff, n_embd_out, n_expert);
        ggml_tensor * cur       = ggml_new_tensor_3d(result.ctx.get(), GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor * ids       = ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_I32, n_expert_used, 1);


        result.out = ggml_moe_ffn_up_gate_qwen35_decode(result.ctx.get(), up_gate_exps, down_exps, cur, ids);
       
        ggml_set_output(result.out);

        ggml_backend_buffer_type_t weight_buft = ggml_backend_cpu_repack_buffer_type();
        result.weight_buffer.reset(ggml_backend_buft_alloc_buffer(weight_buft,
                ggml_nbytes(up_gate_exps) + 3*ggml_backend_buft_get_alignment(weight_buft)));
        ggml_backend_buffer_type_t host_buft = ggml_backend_cpu_buffer_type();
        result.host_buffer.reset(ggml_backend_buft_alloc_buffer(host_buft, buffer_size_for_tensors(host_buft, result.ctx.get(), up_gate_exps, nullptr)));
        if (!result.weight_buffer || !result.host_buffer) {
            fprintf(stderr, "failed to allocate tensor buffers\n");
            std::exit(1);
        }

        ggml_tallocr weight_alloc = ggml_tallocr_new(result.weight_buffer.get());
        ggml_tallocr host_alloc   = ggml_tallocr_new(result.host_buffer.get());

        ggml_tallocr_alloc(&weight_alloc, up_gate_exps);

        for (ggml_tensor * tensor = ggml_get_first_tensor(result.ctx.get()); tensor != nullptr; tensor = ggml_get_next_tensor(result.ctx.get(), tensor)) {
            if (tensor != up_gate_exps && tensor->data == nullptr && tensor->view_src == nullptr) {
                ggml_tallocr_alloc(&host_alloc, tensor);
            }
        }

        set_quantized_tensor(up_gate_exps, up_gate_q);
        set_quantized_tensor(down_exps, down_q);
        ggml_backend_tensor_set(cur, cur_data.data(), 0, cur_data.size()*sizeof(float));
        ggml_backend_tensor_set(ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));

        result.graph = ggml_new_graph_custom(result.ctx.get(), 64, false);
        ggml_build_forward_expand(result.graph, result.out);

        result.backend.reset(ggml_backend_cpu_init());
        ggml_backend_cpu_set_n_threads(result.backend.get(), n_threads);
        return result;
    }
}

static bool nearly_equal(float a, float b) {
    const float diff = std::fabs(a - b);
    const float scale = std::max({1.0f, std::fabs(a), std::fabs(b)});
    return diff <= 2.0e-3f*scale;
}

static bool test_down_type(enum ggml_type down_type, moe_path path) {
    const std::vector<float> up_f32   = make_f32_3d(n_embd, n_ff, n_expert, 1.7f);
    const std::vector<float> gate_f32 = make_f32_3d(n_embd, n_ff, n_expert, 1.7f);
    const std::vector<float> down_f32 = make_f32_3d(n_ff, n_embd_out, n_expert, 2.9f);

    const std::vector<uint8_t> up_q   = quantize_3d(GGML_TYPE_Q4_K, up_f32, n_embd, n_ff, n_expert);
    const std::vector<uint8_t> gate_q = quantize_3d(GGML_TYPE_Q4_K, gate_f32, n_embd, n_ff, n_expert);
    const std::vector<float> up_gate_f32 = make_up_gate_f32_3d(up_f32, gate_f32);
    const std::vector<uint8_t> up_gate_q = quantize_3d(GGML_TYPE_Q4_K, up_gate_f32, 2*n_embd, n_ff, n_expert);
    const std::vector<uint8_t> down_q = quantize_3d(down_type, down_f32, n_ff, n_embd_out, n_expert);

    std::vector<float> cur_data(n_embd);
    for (int64_t i = 0; i < n_embd; ++i) {
        cur_data[i] = (static_cast<float>((i*7) % 29) - 14.0f)*0.02f;
    }

    const std::vector<int32_t> ids_data = { 3, 7, 0, 11, 5, 9, 2, 14 };

    moe_graph unfused_graph       = build_moe_graph(down_type, moe_path::unfused,       up_q, gate_q, up_gate_q, down_q, cur_data, ids_data);
    moe_graph compare_graph         = build_moe_graph(down_type, path,         up_q, gate_q, up_gate_q, down_q, cur_data, ids_data);

    const run_result unfused       = unfused_graph.run();
    const run_result fused         = compare_graph.run();

    bool ok = true;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double sum_sq_diff  = 0.0;
    size_t max_diff_idx = 0;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < unfused.output.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(unfused.output[i]) - fused.output[i]);
        const double rel  = diff / std::max({1.0, std::fabs(static_cast<double>(unfused.output[i])), std::fabs(static_cast<double>(fused.output[i]))});

        sum_sq_diff += diff*diff;
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_diff_idx = i;
        }
        max_rel_diff = std::max(max_rel_diff, rel);

        if (!nearly_equal(unfused.output[i], fused.output[i])) {
            ++mismatch_count;
            ok = false;
        }
    }

    const double rmse = std::sqrt(sum_sq_diff / unfused.output.size());

    printf("%s fused diff: max_abs=%g max_rel=%g rmse=%g mismatches=%zu max_idx=%zu unfused=%g fused=%g\n",
            ggml_type_name(down_type), max_abs_diff, max_rel_diff, rmse, mismatch_count, max_diff_idx,
            unfused.output[max_diff_idx], fused.output[max_diff_idx]);


    if (mismatch_count > 0) {
        fprintf(stderr, "%s mismatch count: %zu\n", ggml_type_name(down_type), mismatch_count);
    }
    
    printf("%s unfused: %8.2f us/run, %d runs\n", ggml_type_name(down_type), unfused.time_us, unfused.n_runs);
    printf("%s fused:   %8.2f us/run, %d runs, speedup %.2fx\n", ggml_type_name(down_type), fused.time_us, fused.n_runs, unfused.time_us/fused.time_us);

    const double phase_runs = std::max<int64_t>(1, fused.timings.calls);
    printf("%s fused phases: cur_q=%6.2f gate_up=%6.2f barrier=%6.2f act_q=%6.2f down=%6.2f us/run, calls=%" PRId64 "\n",
            ggml_type_name(down_type),
            fused.timings.cur_q_us/phase_runs,
            fused.timings.gate_up_us/phase_runs,
            fused.timings.barrier_us/phase_runs,
            fused.timings.act_q_us/phase_runs,
            fused.timings.down_us/phase_runs,
            fused.timings.calls);

    return ok;
}

int main() {
    bool ok = true;
    ok = test_down_type(GGML_TYPE_Q5_K, moe_path::fused) && ok;

    printf("fused test-moe-ffn-qwen35-decode: OK\n");

    ok = test_down_type(GGML_TYPE_Q5_K, moe_path::gate_up_fused) && ok;
    printf("gate up fused test-moe-ffn-qwen35-decode: OK\n");
    return 0;
}
