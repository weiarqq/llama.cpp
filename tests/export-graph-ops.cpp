#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "ggml.h"
#include "gguf-model-data.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "download.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Noop because weights are not needed
static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
}

static std::string json_escape(const char * value) {
    std::ostringstream out;
    out << '"';
    if (value != nullptr) {
        for (const unsigned char c : std::string(value)) {
            switch (c) {
                case '\\': out << "\\\\"; break;
                case '"':  out << "\\\""; break;
                case '\b': out << "\\b";  break;
                case '\f': out << "\\f";  break;
                case '\n': out << "\\n";  break;
                case '\r': out << "\\r";  break;
                case '\t': out << "\\t";  break;
                default:
                    if (c < 0x20) {
                        out << "\\u" << std::hex << std::uppercase << std::setw(4) << std::setfill('0') << (int) c
                            << std::dec << std::nouppercase << std::setfill(' ');
                    } else {
                        out << c;
                    }
            }
        }
    }
    out << '"';
    return out.str();
}

static std::filesystem::path normalized_output_path(const std::string & path, std::error_code & ec) {
    namespace fs = std::filesystem;

    fs::path p = fs::absolute(path, ec);
    if (ec) {
        return {};
    }

    const fs::file_status status = fs::symlink_status(p, ec);
    if (ec) {
        if (ec != std::errc::no_such_file_or_directory) {
            return {};
        }
        ec.clear();
    }

    if (fs::is_symlink(status)) {
        fs::path target = fs::read_symlink(p, ec);
        if (ec) {
            return {};
        }
        if (target.is_relative()) {
            target = p.parent_path() / target;
        }
        p = target;
    }

    ec.clear();
    return fs::weakly_canonical(p, ec).lexically_normal();
}

static bool same_output_path(const std::string & a, const std::string & b) {
    namespace fs = std::filesystem;

    if (a == b) {
        return true;
    }

    std::error_code ec_a;
    std::error_code ec_b;
    if (fs::exists(a, ec_a) && fs::exists(b, ec_b) && fs::equivalent(a, b, ec_a)) {
        return true;
    }

    ec_a.clear();
    ec_b.clear();
    const fs::path norm_a = normalized_output_path(a, ec_a);
    const fs::path norm_b = normalized_output_path(b, ec_b);
    if (ec_a || ec_b) {
        return false;
    }

    return norm_a == norm_b;
}

static void write_i64_array(std::ostream & out, const int64_t * values, int n) {
    out << '[';
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

static void write_size_array(std::ostream & out, const size_t * values, int n) {
    out << '[';
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

struct input_tensor {
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::array<size_t, 4> nb;

    input_tensor(ggml_type type, int64_t * ne, size_t * nb): type(type) {
        memcpy(this->ne.data(), ne, 4 * sizeof(int64_t));
        memcpy(this->nb.data(), nb, 4 * sizeof(size_t));
    }

    bool operator<(const input_tensor &b) const {
        return std::tie(type, ne, nb) <
               std::tie(b.type, b.ne, b.nb);
    }

    void serialize(std::ostream& out) const {
        out << type << ' ';
        for (size_t i = 0; i < 4; i++) {
            out << ne[i] << ' ';
        }
        for (size_t i = 0; i < 4; i++) {
            out << nb[i] << ' ';
        }
    }
};

struct test_object {
    ggml_op op;
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::vector<int32_t> op_params;
    std::vector<input_tensor> sources;
    std::string name;

    void serialize(std::ostream& out) const {
        out << op << ' ' << type << ' ';
        for (size_t i = 0; i < 4; i++) {
            out << ne[i] << ' ';
        }

        out << op_params.size() << ' ';
        for (size_t i = 0; i < op_params.size(); i++) {
            out << op_params[i] << ' ';
        }

        out << sources.size() << ' ';
        for (size_t s = 0; s < sources.size(); s++) {
            sources[s].serialize(out);
        }

        if (!name.empty()) {
            out << name;
        } else {
            out << '-';
        }

        out << '\n';
    }

    bool operator<(const test_object &b) const {
        return std::tie(op, type, ne, op_params, sources) <
               std::tie(b.op, b.type, b.ne, b.op_params, b.sources);
    }
};

static void extract_graph_ops(ggml_cgraph * cgraph, const char * label, std::set<test_object> & tests) {
    int n_nodes = ggml_graph_n_nodes(cgraph);
    int n_skipped = 0;
    int n_before = (int) tests.size();
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);

        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            n_skipped++;
            continue;
        }

        test_object test;

        test.op = node->op;
        test.type = node->type;
        memcpy(&test.ne, node->ne, 4 * sizeof(int64_t));

        test.op_params.resize(GGML_MAX_OP_PARAMS / sizeof(int32_t));
        memcpy(test.op_params.data(), node->op_params, GGML_MAX_OP_PARAMS);

        for (size_t s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s] == nullptr) {
                break;
            }

            test.sources.emplace_back(node->src[s]->type, node->src[s]->ne, node->src[s]->nb);
        }

        test.name = node->name;
        tests.insert(test);
    }

    int n_new = (int) tests.size() - n_before;
    LOG_INF("%s: %d unique ops, %d total nodes, %d skipped (view ops)\n",
            label, n_new, n_nodes, n_skipped);
}

static void write_graph_json(std::ostream & out, ggml_cgraph * cgraph, const char * label, bool first_graph) {
    std::map<const ggml_tensor *, int> node_ids;
    const int n_nodes = ggml_graph_n_nodes(cgraph);

    for (int i = 0; i < n_nodes; ++i) {
        node_ids[ggml_graph_node(cgraph, i)] = i;
    }

    if (!first_graph) {
        out << ",\n";
    }

    out << "    {\n";
    out << "      \"label\": " << json_escape(label) << ",\n";
    out << "      \"nodes\": [\n";

    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);

        if (i > 0) {
            out << ",\n";
        }

        out << "        {";
        out << "\"id\": " << i;
        out << ", \"name\": " << json_escape(ggml_get_name(node));
        out << ", \"op\": " << json_escape(ggml_op_name(node->op));
        out << ", \"type\": " << json_escape(ggml_type_name(node->type));
        out << ", \"ne\": ";
        write_i64_array(out, node->ne, GGML_MAX_DIMS);
        out << ", \"nb\": ";
        write_size_array(out, node->nb, GGML_MAX_DIMS);
        out << ", \"src\": [";
        bool first_src = true;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] == nullptr) {
                break;
            }
            const auto it = node_ids.find(node->src[s]);
            if (it == node_ids.end()) {
                continue;
            }
            if (!first_src) {
                out << ", ";
            }
            out << it->second;
            first_src = false;
        }
        out << "]}";
    }

    out << "\n      ],\n";
    out << "      \"edges\": [\n";

    bool first_edge = true;
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] == nullptr) {
                break;
            }
            const auto it = node_ids.find(node->src[s]);
            if (it == node_ids.end()) {
                continue;
            }
            if (!first_edge) {
                out << ",\n";
            }
            out << "        {\"from\": " << it->second << ", \"to\": " << i << ", \"slot\": " << s << "}";
            first_edge = false;
        }
    }

    out << "\n      ]\n";
    out << "    }";
}

static void write_graph_json_header(std::ostream & out) {
    out << "{\n";
    out << "  \"graphs\": [\n";
}

static void write_graph_json_footer(std::ostream & out) {
    out << "\n  ]\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "tests.txt";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS)) {
        return 1;
    }

    if (!params.graph_out_file.empty() && same_output_path(params.graph_out_file, params.out_file)) {
        LOG_ERR("graph JSON output file must differ from output file: %s\n", params.graph_out_file.c_str());
        return 1;
    }

    // Load CPU-only
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    params.devices = { cpu_device, nullptr };
    params.fit_params = false;
    params.n_gpu_layers = 0;

    params.warmup = false;

    llama_context * ctx;
    common_init_result_ptr init_result;
    llama_context_ptr ctx2;
    llama_model_ptr model;

    if (params.model.hf_repo.empty()) {
        init_result = common_init_from_params(params);

        ctx = init_result->context();
    } else {
#ifdef LLAMA_HF_FETCH
        auto [hf_repo, hf_quant] = common_download_split_repo_tag(params.model.hf_repo);
        if (hf_quant.empty() || hf_quant == "latest") {
            hf_quant = "Q4_K_M";
        }

        gguf_context_ptr gguf_ctx = gguf_fetch_gguf_ctx(hf_repo, hf_quant);
        if (!gguf_ctx) {
            LOG_ERR("failed to fetch GGUF metadata from %s\n", hf_repo.c_str());
            return 1;
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.devices = params.devices.data();
        model_params.no_alloc = true;

        model.reset(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, nullptr, model_params));

        if (!model) {
            LOG_ERR("failed to create llama_model from %s\n", hf_repo.c_str());
            return 1;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx2.reset(llama_init_from_model(model.get(), ctx_params));
        ctx = ctx2.get();

        if (!ctx) {
            LOG_ERR("failed to create llama_context\n");
            return 1;
        }
#else
        LOG_ERR("export-graph-ops compiled without HF fetch support\n");
        return 1;
#endif
    }

    const uint32_t n_seqs  = llama_n_seq_max(ctx);
    const uint32_t n_tokens = std::min(llama_n_ctx(ctx), llama_n_ubatch(ctx));

    std::set<test_object> tests;
    std::ofstream graph_json;

    if (!params.graph_out_file.empty()) {
        graph_json.open(params.graph_out_file);
        if (!graph_json.is_open()) {
            LOG_ERR("unable to open graph JSON output file: %s\n", params.graph_out_file.c_str());
            return 1;
        }
        write_graph_json_header(graph_json);
    }

    auto * gf_pp = llama_graph_reserve(ctx, n_tokens, n_seqs, n_tokens);
    if (!gf_pp) {
        LOG_ERR("failed to reserve prompt processing graph\n");
        return 1;
    }
    extract_graph_ops(gf_pp, "pp", tests);
    if (graph_json.is_open()) {
        write_graph_json(graph_json, gf_pp, "pp", true);
    }

    auto * gf_tg = llama_graph_reserve(ctx, n_seqs, n_seqs, n_seqs);
    if (!gf_tg) {
        LOG_ERR("failed to reserve token generation graph\n");
        return 1;
    }
    extract_graph_ops(gf_tg, "tg", tests);

    if (graph_json.is_open()) {
        write_graph_json(graph_json, gf_tg, "tg", false);
        write_graph_json_footer(graph_json);
        graph_json.flush();
        if (!graph_json) {
            LOG_ERR("failed to write graph JSON output file: %s\n", params.graph_out_file.c_str());
            return 1;
        }
        LOG_INF("wrote graph JSON to %s\n", params.graph_out_file.c_str());
    }

    LOG_INF("%d unique ops total\n", (int) tests.size());

    std::ofstream f(params.out_file);

    if (!f.is_open()) {
        LOG_ERR("unable to open output file: %s\n", params.out_file.c_str());
        return 1;
    }

    for (const auto& test : tests) {
        test.serialize(f);
    }

    return 0;
}
