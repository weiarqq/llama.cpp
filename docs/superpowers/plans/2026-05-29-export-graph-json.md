# Export Graph JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional JSON export to `tests/export-graph-ops.cpp` that emits complete prompt-processing and token-generation computation graphs for visualization.

**Architecture:** Keep the existing unique-op text export unchanged. Add a separate `--graph-json <file>` CLI option and a small JSON serializer in `tests/export-graph-ops.cpp` that walks each `ggml_cgraph`, records nodes with stable integer ids, and emits edges from each source tensor to its consumer when the source is also a graph node.

**Tech Stack:** C++17, llama.cpp common argument parser, `ggml_cgraph` / `ggml_tensor`, `ggml_op_name`, `ggml_type_name`, `std::ofstream`.

---

## File Structure

- Modify `common/common.h`: add a `std::string graph_out_file` field to `common_params` near `out_file` so common argument parsing can store the JSON path.
- Modify `common/arg.cpp`: register `--graph-json FNAME` only for `LLAMA_EXAMPLE_EXPORT_GRAPH_OPS`.
- Modify `tests/export-graph-ops.cpp`: add JSON escaping, tensor shape helpers, graph JSON serialization, and call it for `pp` and `tg` graphs when `params.graph_out_file` is non-empty.

Do not change the existing `tests.txt` unique-op output format.

### Task 1: Add CLI Storage And Option

**Files:**
- Modify: `common/common.h`
- Modify: `common/arg.cpp`

- [ ] **Step 1: Add a parameter field**

In `common/common.h`, find `std::string out_file; // output filename for all example programs` in `struct common_params` and add this field immediately after it:

```cpp
std::string graph_out_file; // output filename for graph JSON exports
```

- [ ] **Step 2: Add the CLI option**

In `common/arg.cpp`, immediately after the existing `-o`, `--output`, `--output-file` option block for `params.out_file`, add:

```cpp
add_opt(common_arg(
    {"--graph-json"}, "FNAME",
    "output computation graph JSON file",
    [](common_params & params, const std::string & value) {
        params.graph_out_file = value;
    }
).set_examples({LLAMA_EXAMPLE_EXPORT_GRAPH_OPS}));
```

- [ ] **Step 3: Build the target to catch parser errors**

Run:

```bash
cmake --build build --target export-graph-ops
```

Expected: build succeeds, or fails only if the local build directory does not exist. If `build` does not exist, use the repository's existing build directory if present; do not create a new full build unless the user approves.

### Task 2: Add JSON Serialization Helpers

**Files:**
- Modify: `tests/export-graph-ops.cpp`

- [ ] **Step 1: Add required includes**

At the top of `tests/export-graph-ops.cpp`, add these includes with the existing standard-library includes:

```cpp
#include <map>
#include <sstream>
```

- [ ] **Step 2: Add JSON helper functions**

After `set_tensor_data`, add:

```cpp
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
```

Also add `#include <iomanip>` if it is not already present, because `json_escape` uses `std::setw` and `std::setfill`.

- [ ] **Step 3: Build to catch helper issues**

Run:

```bash
cmake --build build --target export-graph-ops
```

Expected: build succeeds, or fails only because no usable build directory exists.

### Task 3: Serialize Complete Graphs To JSON

**Files:**
- Modify: `tests/export-graph-ops.cpp`

- [ ] **Step 1: Add graph JSON writer**

After `extract_graph_ops`, add:

```cpp
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
```

- [ ] **Step 2: Add top-level JSON file writer**

After `write_graph_json`, add:

```cpp
static bool export_graph_json(const std::string & filename, ggml_cgraph * gf_pp, ggml_cgraph * gf_tg) {
    std::ofstream f(filename);

    if (!f.is_open()) {
        LOG_ERR("unable to open graph JSON output file: %s\n", filename.c_str());
        return false;
    }

    f << "{\n";
    f << "  \"graphs\": [\n";
    write_graph_json(f, gf_pp, "pp", true);
    write_graph_json(f, gf_tg, "tg", false);
    f << "\n  ]\n";
    f << "}\n";

    return true;
}
```

- [ ] **Step 3: Build to catch serializer errors**

Run:

```bash
cmake --build build --target export-graph-ops
```

Expected: build succeeds, or fails only because no usable build directory exists.

### Task 4: Wire JSON Export Into Main

**Files:**
- Modify: `tests/export-graph-ops.cpp`

- [ ] **Step 1: Call JSON export after both graphs are reserved**

In `main`, immediately after:

```cpp
extract_graph_ops(gf_tg, "tg", tests);
```

add:

```cpp
if (!params.graph_out_file.empty()) {
    if (!export_graph_json(params.graph_out_file, gf_pp, gf_tg)) {
        return 1;
    }
    LOG_INF("wrote graph JSON to %s\n", params.graph_out_file.c_str());
}
```

- [ ] **Step 2: Build the target**

Run:

```bash
cmake --build build --target export-graph-ops
```

Expected: build succeeds, or fails only because no usable build directory exists.

### Task 5: Verify Output Shape

**Files:**
- Verify: built `export-graph-ops` executable

- [ ] **Step 1: Run help and confirm option is shown**

Run the built executable with help. Use the path that exists in the local build tree, for example:

```bash
./build/bin/export-graph-ops --help
```

Expected: output includes `--graph-json FNAME`.

- [ ] **Step 2: Run a small graph export if a model is available**

If the workspace has a small GGUF model available, run:

```bash
./build/bin/export-graph-ops -m /path/to/model.gguf -c 16 -b 4 -ub 4 -o /tmp/export-graph-ops.txt --graph-json /tmp/export-graph.json
```

Expected: command exits 0 and writes both `/tmp/export-graph-ops.txt` and `/tmp/export-graph.json`.

- [ ] **Step 3: Validate JSON syntax**

Run:

```bash
python3 -m json.tool /tmp/export-graph.json >/tmp/export-graph.pretty.json
```

Expected: command exits 0.

- [ ] **Step 4: Check graph content**

Run:

```bash
python3 - <<'PY'
import json
with open('/tmp/export-graph.json') as f:
    data = json.load(f)
labels = [g['label'] for g in data['graphs']]
assert labels == ['pp', 'tg'], labels
for graph in data['graphs']:
    assert graph['nodes'], graph['label']
    assert all('id' in n and 'op' in n and 'type' in n and 'ne' in n and 'nb' in n and 'src' in n for n in graph['nodes'])
    node_ids = {n['id'] for n in graph['nodes']}
    for edge in graph['edges']:
        assert edge['from'] in node_ids
        assert edge['to'] in node_ids
        assert isinstance(edge['slot'], int)
print('ok')
PY
```

Expected: prints `ok`.

## Self-Review

- Spec coverage: The plan preserves existing unique-op export, adds optional JSON via `--graph-json`, emits `pp` and `tg`, includes node metadata, and includes explicit edges for visualization.
- Placeholder scan: No implementation step uses TBD/TODO/fill-in placeholders. The only conditional is model availability for runtime verification.
- Type consistency: The new parameter is consistently named `graph_out_file`; the CLI option writes it; `main` reads it; serializer signatures use `ggml_cgraph *` consistently.
