# Qwen3.5-MoE SYCL Intel Graph Optimization Analysis

Source graph: `qwen3.5-moe-graph.json`

Layer-grouped graph: `qwen3.5-moe-graph-by-layer.json`

Target observation from Intel optimized SYCL build:

```text
sched_reserve: graph nodes = 966 (with bs=512), 906 (with bs=1)
```

Upstream local observation:

```text
sched_reserve: graph nodes = 3729
```

## Interpretation

If Intel prints `966/906` from `sched_reserve`, then their optimization is probably not only a runtime skip inside `ggml_backend_sycl_graph_compute_impl()`. The node count printed by llama.cpp comes from `ggml_graph_n_nodes(gf)` during reservation in `src/llama-context.cpp`, so a smaller number means the graph has already been rewritten or compacted before the final reserved graph is returned.

The upstream scheduler has a hook for this exact point:

```cpp
// ggml/src/ggml-backend.cpp
split->graph = ggml_graph_view(graph, split->i_start, split->i_end);
ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);
```

In current upstream SYCL, the callback is not implemented:

```cpp
// ggml/src/ggml-sycl/ggml-sycl.cpp
/*.graph_optimize = */ NULL,
```

Therefore the most likely Intel change is a private SYCL `graph_optimize` implementation that rewrites each scheduler split before reservation accounting and allocation.

## Original Graph Accounting

For each of `pp` and `tg` in the exported graph:

```text
total nodes: 3729

layout/meta nodes:
VIEW       792
RESHAPE    520
TRANSPOSE   30
PERMUTE     30
total     1372

non-layout nodes:         2357
non-layout, non-empty:    2237
non-layout, non-empty, excluding SCALE: 2177
```

Layer breakdown:

```text
30 linear attention layers: ~99 nodes/layer
10 full attention layers:   ~75 nodes/layer
global before: 1 node
global after:  3 nodes
```

The Intel numbers match this per-layer shape closely:

```text
pp: 30 * 27 + 10 * 15 + 6 = 966
tg: 30 * 25 + 10 * 15 + 6 = 906
```

This strongly suggests Intel reduced each linear attention layer to about `27` nodes in prompt processing, about `25` nodes in token generation, and each full attention layer to about `15` nodes. The `pp - tg = 60` difference equals `30 linear layers * 2 nodes`, so the difference is probably localized to the linear attention path.

## Likely Optimization Passes

### 1. Structural Layout Elimination

Upstream SYCL already skips these at compute time:

```cpp
GGML_OP_RESHAPE
GGML_OP_TRANSPOSE
GGML_OP_VIEW
GGML_OP_PERMUTE
GGML_OP_NONE
```

Intel likely moves this earlier into graph optimization, physically removing layout nodes from the graph and reconnecting users to their real data sources with updated tensor views/strides.

This explains a large part of the reduction:

```text
3729 - 1372 = 2357
```

But `2357` is still much larger than `966/906`, so layout elimination alone is not enough.

### 2. Empty Tensor and Dead Cache Branch Pruning

The linear attention layers contain zero-sized history/cache paths, for example:

```text
GET_ROWS [24576, 0, 1, 1]
CPY      [24576, 0, 1, 1]
GET_ROWS [524288, 0, 1, 1]
CPY      [524288, 0, 1, 1]
```

These occur in the linear attention recurrent cache path. Upstream SYCL skips `ggml_is_empty(node)` at compute time. Intel likely prunes these nodes at graph optimization time when the shapes are known.

This removes about `181` zero-sized nodes from the 30 linear layers in this exported graph.

### 3. RMS_NORM + MUL Fusion

CPU already has this fusion as a precedent:

```cpp
RMS_NORM + MUL
```

Qwen3.5-MoE repeatedly emits this pair as:

```text
norm-N -> attn_norm-N
norm-N -> attn_post_norm-N
norm inside q/k normalization blocks
```

Intel likely compacts those pairs into one graph node or into a single SYCL kernel invocation. This directly reduces both linear and full attention layers.

### 4. Gated Delta Net Path Fusion

The linear attention core is already expressed as the high-level op:

```text
GATED_DELTA_NET
```

But the surrounding path is still verbose in upstream ggml:

```text
cache_r read/update
CONCAT
SSM_CONV
SILU
q/k/v VIEWs
L2_NORM q/k
alpha/beta projections and activations
GATED_DELTA_NET
new_state VIEW/CPY
z gate + norm + output projection
```

Likely Intel fusions:

```text
SSM_CONV + SILU + q/k/v slicing
L2_NORM folded into GDN input preparation
alpha ADD + SOFTPLUS + MUL folded into gate preparation
beta SIGMOID folded into beta preparation
GATED_DELTA_NET + new_state extraction/update
RMS_NORM + z activation + MUL folded around linear_attn_out
```

The `tg` path is likely even smaller because `n_seq_tokens == 1` uses the autoregressive GDN route. The observed difference is exactly `2 nodes/layer` across the 30 linear layers:

```text
966 - 906 = 60 = 30 * 2
```

Likely `tg` removes two chunked-path nodes per linear layer, such as chunk-state handling or extra recurrent cache update/slice nodes.

### 5. Full Attention Compaction Around FLASH_ATTN_EXT

Full attention layers already use:

```text
FLASH_ATTN_EXT
```

The upstream graph still includes many surrounding nodes:

```text
Q/K/V projections
Q/K RMS_NORM + MUL
ROPE
SET_ROWS cache updates
PERMUTE/VIEW layout nodes
FLASH_ATTN_EXT
gate sigmoid + multiply
output projection
residual add
MoE/shared expert FFN
```

Likely Intel fusions:

```text
Q/K RMS_NORM + MUL + ROPE folded into Q/K preparation
cache K/V SET_ROWS fused or compacted with K/V projection path
FLASH_ATTN_EXT kept as one high-level op
attention gate sigmoid + MUL folded into post-attention kernel
residual ADD fused into output projection or post-op
```

This makes about `15` nodes per full attention layer plausible.

### 6. MoE Router and Expert Aggregation Fusion

Every layer contains the same MoE FFN tail:

```text
ffn_moe_logits      MUL_MAT
ffn_moe_probs       SOFT_MAX
ffn_moe_argsort     ARGSORT
ffn_moe_weights     GET_ROWS
ffn_moe_weights_sum SUM_ROWS
CLAMP
DIV
ffn_moe_gate        MUL_MAT_ID
ffn_moe_up          MUL_MAT_ID
GLU
ffn_moe_down        MUL_MAT_ID
weighted MUL
7 ADDs to reduce top-k experts
shared expert gate/up/down path
shared expert sigmoid/mul
final ADDs
```

Likely Intel fusions:

```text
SOFT_MAX + ARGSORT + GET_ROWS + SUM_ROWS + CLAMP + DIV -> router/topk-normalize node
MUL_MAT_ID gate/up + GLU -> fused expert up/gate node or two matmuls with fused activation
MUL_MAT_ID down + expert weights + 7 ADD reductions -> fused weighted expert reduction
shared expert sigmoid + MUL -> fused shared expert gate
final ffn_out/l_out ADDs -> fused residual post-op
```

This is likely the second major source of reduction after layout pruning.

## Plausible Reproduction Strategy

The safest way to reproduce Intel's behavior is not to start by changing Qwen graph construction. Instead, add a backend graph optimization pass for SYCL, because the scheduler already calls this hook before the reserved graph is copied and allocated.

Recommended sequence:

1. Add `ggml_backend_sycl_graph_optimize()` and register it in the SYCL backend interface.
2. First implement analysis-only logging inside the optimizer: before/after node counts, op histograms, per-layer counts by tensor name.
3. Implement graph compaction for pure metadata ops: `VIEW`, `RESHAPE`, `TRANSPOSE`, `PERMUTE`, `NONE` where users can safely be rewired.
4. Prune empty nodes and nodes with `GGML_TENSOR_FLAG_COMPUTE == 0` when safe.
5. Add pattern fusions guarded by exact op/name/shape checks for Qwen3.5-MoE first.
6. Validate after each pass against the target counts:

```text
bs=512 target: 966
bs=1   target: 906
```

Avoid attempting all fusions at once. The first milestone should be to reproduce count deltas, not performance.

## Candidate Per-Layer Optimized Node Budget

This is a working hypothesis that matches Intel's printed counts.

### Linear Attention, PP: about 27 nodes

Likely retained/fused units:

```text
input RMS_NORM/MUL                    1
qkvz projection                       1
alpha projection + activation         1-2
beta projection + sigmoid             1-2
conv state read/update                1-2
SSM_CONV/SILU/qkv split               1-2
q/k normalization                     1-2
GATED_DELTA_NET + state update        1-2
z projection/gate/norm                2-3
linear_attn_out + residual            1-2
MoE router/topk normalize             1-2
MoE gate/up/down expert path          3-5
expert weighted reduction             1
shared expert path                    2-3
final residual                        1
```

### Linear Attention, TG: about 25 nodes

Same as PP, but autoregressive GDN likely removes about two chunk/cache handling nodes per layer.

### Full Attention: about 15 nodes

Likely retained/fused units:

```text
input RMS_NORM/MUL                    1
Q/K/V preparation                     2-3
Q/K norm + ROPE + cache update        2-3
FLASH_ATTN_EXT                        1
attention gate/output/residual        2-3
MoE router/topk normalize             1
MoE expert path/reduction             3-4
shared expert/final residual          1-2
```

## Important Caveats

This is an inference from graph shapes and public upstream code, not a reconstruction from Intel source. The exact Intel implementation may use different internal fused nodes, may count copied scheduler inputs differently, or may combine graph optimization with a custom allocator/meta backend.

The strongest evidence is the arithmetic match:

```text
30 linear * 27 + 10 full * 15 + 6 = 966
30 linear * 25 + 10 full * 15 + 6 = 906
```

The weakest part is the exact internal boundary of each fused pattern. That should be validated by instrumenting a local SYCL `graph_optimize` pass and comparing intermediate counts after each pass.

## Addendum: Intel Log Says Flash Attention Auto Disabled

Additional Intel optimized build log:

```text
Flash Attention was auto, set to disabled
sched_reserve: graph nodes = 966 (with bs=512), 906 (with bs=1)
```

This changes the interpretation of the full-attention part. In upstream llama.cpp, this log is emitted from `src/llama-context.cpp` when the auto-FA probe builds a temporary graph containing `GGML_OP_FLASH_ATTN_EXT`, then finds the FA tensor assigned to a different device than the model layer/KV device:

```cpp
if (device_fa != device_kv) {
    fa_device_mismatch = true;
}

if (fa_device_mismatch) {
    cparams.flash_attn = false;
    LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
}
```

That means the final reserved `pp/tg` graph in the Intel run probably does **not** contain upstream `FLASH_ATTN_EXT` nodes. Instead, the full attention layers are likely built through the non-FA path:

```text
kq = MUL_MAT(K, Q)
kq = SOFT_MAX_EXT(kq, mask)
kqv = MUL_MAT(V, kq)
layout recombination
```

Therefore the earlier phrase "full attention compaction around FLASH_ATTN_EXT" should be read as a weaker hypothesis for the local exported graph, not as the likely Intel optimized path. With FA disabled, the more likely Intel implementation is:

```text
non-FA attention subgraph fusion:
Q/K/V prep + KQ matmul + mask/softmax + V*KQ matmul + output layout
```

Possible reasons Intel disables upstream FA but still gets a small optimized graph:

1. `GGML_OP_FLASH_ATTN_EXT` is unsupported or assigned to CPU in their SYCL device check, so auto-FA disables it.
2. Intel has a separate graph optimizer that recognizes the non-FA attention pattern and replaces it with one or a few optimized SYCL nodes.
3. Their optimized attention path may be implemented as a graph fusion pass rather than exposed as upstream `GGML_OP_FLASH_ATTN_EXT`.
4. Disabling upstream FA may be intentional because their non-FA fused pattern is faster or easier to compile for their hardware/compiler stack.

This makes the target count even more informative: reaching `966/906` with upstream FA disabled implies their graph optimizer is stronger than simple layout pruning and stronger than relying on existing ggml high-level FA ops. It likely performs at least one full-attention subgraph replacement pass.

Updated likely full-attention optimization:

```text
Original upstream FA-enabled exported full layer: ~75 nodes
Intel FA-disabled optimized full layer: still likely ~15 nodes

Therefore Intel probably fuses the non-FA attention chain to an attention supernode or compact set of nodes.
```

To reproduce this more faithfully, generate a second reference graph with Flash Attention disabled:

```bash
./build/bin/export-graph-ops \
  -m /path/to/qwen3.5-moe.gguf \
  -c 512 -b 512 -ub 512 \
  -fa off \
  -o /tmp/qwen35moe-no-fa-ops.txt \
  --graph-json /tmp/qwen35moe-no-fa-graph.json
```

Then compare:

```text
FA-enabled local graph:    contains __fattn__ / FLASH_ATTN_EXT
FA-disabled local graph:   contains kq / kq_soft_max / kqv path
Intel optimized graph:     FA disabled, but count still 966/906
```

The implementation target should therefore include both classes of fusion:

```text
linear attention / GDN fusion
MoE router and expert fusion
non-FA full-attention pattern fusion
layout and empty-node pruning
```

## Addendum: Intel Log Does Not Show Fused GDN Auto-Detection

Additional Intel optimized build observation: these upstream logs are absent:

```text
sched_reserve: resolving fused Gated Delta Net support:
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
```

In current upstream llama.cpp, these logs are unconditional once `cparams.auto_fgdn` is true:

```cpp
cparams.fused_gdn_ar = true;
cparams.fused_gdn_ch = true;
cparams.auto_fgdn    = true;

if (cparams.auto_fgdn) {
    LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);
    ...
    LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
    ...
    LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
}
```

Therefore, if the Intel binary prints the FA auto-disable log but does not print the GDN auto-detection logs, the Intel tree is not following this upstream control flow exactly. Likely explanations:

1. Intel removed the `auto_fgdn` probing block and directly sets final GDN behavior.
2. Intel initializes `auto_fgdn = false` earlier, so the probe block is skipped.
3. Intel renamed or suppressed these log lines while keeping fused GDN enabled.
4. Intel no longer relies on `GGML_OP_GATED_DELTA_NET` auto device probing, because their graph optimizer recognizes and fuses the linear attention pattern independently.
5. Intel backported/branched from a version before this auto-GDN logging existed, then added their own graph optimizer.

This weakens any conclusion that depends on seeing `__fgdn_ar__` or `__fgdn_ch__` in Intel's optimized graph. The safer inference is:

```text
Intel optimized graph count 966/906 probably comes from a private graph-level optimizer.
It may use upstream GGML_OP_GATED_DELTA_NET, but that is no longer guaranteed by the logs.
```

The `pp - tg = 60` difference still strongly points to the linear-attention/recurrent path:

```text
966 - 906 = 60 = 30 linear layers * 2 nodes/layer
```

But the cause should be stated more generally as:

```text
tg uses a shorter single-token recurrent linear-attention path than pp,
not necessarily specifically upstream fused_gdn_ar vs fused_gdn_ch.
```

Updated reproduction priority:

1. Generate `-fa off` reference graphs, because Intel disables FA auto.
2. Add local instrumentation to record whether upstream Qwen3.5-MoE builds `GGML_OP_GATED_DELTA_NET` when FA is off.
3. Do not rely on upstream GDN auto-detection logs as a required target; use node names/op histograms instead.
4. Try to match counts with passes ordered as: metadata pruning, empty pruning, non-FA attention fusion, linear-attention recurrent fusion, MoE fusion.

## Addendum: Full Log Comparison Narrows the Hypothesis

Current upstream-like local run:

```text
sched_reserve: reserving ...
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
sched_reserve:      SYCL0 compute buffer size =   804.02 MiB
sched_reserve:  SYCL_Host compute buffer size =   520.02 MiB
sched_reserve: graph nodes  = 3729
sched_reserve: graph splits = 2
```

Intel optimized run:

```text
sched_reserve: reserving ...
sched_reserve: Flash Attention was auto, set to disabled
sched_reserve:      SYCL0 compute buffer size =   172.02 MiB
sched_reserve:  SYCL_Host compute buffer size =   136.02 MiB
sched_reserve: graph nodes  = 966 (with bs=512), 906 (with bs=1)
sched_reserve: graph splits = 2
```

This comparison adds two important constraints:

1. The Intel build is not merely skipping nodes at execution time. The compute buffers shrink from `804/520 MiB` to `172/136 MiB`, which means the allocator/reserve graph is smaller before execution.
2. The graph split count stays `2`, so the optimization likely happens inside each split or before split graph copying, not by changing the high-level scheduler partitioning.

The strongest current hypothesis is:

```text
Intel added a backend-level graph optimizer for SYCL that runs during scheduler reservation.
It compacts or replaces subgraphs before allocation, reducing both node count and buffer size.
```

This matches the upstream scheduler hook:

```cpp
split->graph = ggml_graph_view(graph, split->i_start, split->i_end);
ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);
```

In upstream SYCL this hook is null, so local code keeps the original `3729` nodes and large allocation.

The missing GDN logs imply Intel also changed control flow before normal reservation. They probably either:

```text
disable upstream FA auto path after probing,
skip upstream auto_fgdn probing,
then rely on their graph optimizer to fuse Qwen3.5-MoE-specific patterns.
```

The buffer-size reduction supports the idea that the optimizer removes intermediate tensors, not only node descriptors. Approximate reduction:

```text
SYCL0:     804.02 MiB -> 172.02 MiB, about 78.6% lower
SYCL_Host: 520.02 MiB -> 136.02 MiB, about 73.8% lower
nodes pp:  3729 -> 966, about 74.1% lower
```

The node-count and buffer-size reductions are in the same range, which is consistent with graph compaction that removes large temporary tensors from attention, MoE reduction, and layout/materialization paths.

Updated reproduction target should therefore be measured with three counters, not only `graph nodes`:

```text
graph nodes
SYCL0 compute buffer size
SYCL_Host compute buffer size
```

Matching only `966/906` is insufficient if buffer sizes remain close to `804/520 MiB`.

## Addendum: FA-Disabled Reference Graph

New local files:

```text
qwen35moe-no-fa-ops.txt
qwen35moe-no-fa-graph.json
qwen35moe-no-fa-graph-by-layer.json
```

The `-fa off` graph confirms the Intel FA-disabled path should not contain `FLASH_ATTN_EXT`:

```text
FLASH_ATTN_EXT: 0
GATED_DELTA_NET: 30
kq/kq_soft_max/kqv path: present in full attention layers
```

Per graph (`pp` and `tg` are structurally identical in the exported reserve graph):

```text
total nodes: 3778

layout/meta nodes:
VIEW       782
RESHAPE    540
TRANSPOSE   30
PERMUTE     40
total     1392

non-layout nodes:      2386
non-layout non-empty:  2266
```

Op histogram:

```text
VIEW            782
RESHAPE         540
ADD             430
MUL_MAT         411
MUL             281
UNARY           170
GET_ROWS        163
RMS_NORM        131
CPY             120
MUL_MAT_ID      120
GLU              80
SCALE            60
L2_NORM          60
SOFT_MAX         50
ARGSORT          40
SUM_ROWS         40
CLAMP            40
DIV              40
PERMUTE          40
TRANSPOSE        30
CONCAT           30
SSM_CONV         30
GATED_DELTA_NET  30
ROPE             20
SET_ROWS         20
CONT             20
```

Layer breakdown:

```text
30 linear attention layers: ~99.07 nodes/layer
10 full attention layers:   ~80.2 nodes/layer
global before/after:        4 nodes
```

The full attention layer grows from about `75` nodes in the FA-enabled reference to about `80` nodes in the FA-disabled reference. The extra nodes are the explicit non-FA path:

```text
kq-3           MUL_MAT
kq_soft_max-3  SOFT_MAX
kqv-3          MUL_MAT
additional PERMUTE/CONT/layout nodes
```

This makes the Intel target even more clearly a graph-level fusion result:

```text
no-FA local graph: 3778 nodes
Intel pp target:   966 nodes, reduction = 2812 nodes, 74.4%
Intel tg target:   906 nodes, reduction = 2872 nodes, 76.0%
```

The previous per-layer target arithmetic still matches exactly:

```text
pp: 30 linear * 27 + 10 full * 15 + 6 = 966
tg: 30 linear * 25 + 10 full * 15 + 6 = 906
```

Updated conclusion from the FA-disabled reference:

```text
Intel likely compresses a raw no-FA full attention layer from ~80 nodes to ~15 nodes.
It likely compresses a raw linear attention layer from ~99 nodes to ~27 in pp and ~25 in tg.
```

The presence of `GATED_DELTA_NET: 30` in the local `-fa off` graph also shows that FA disabling does not inherently disable fused GDN in the upstream graph builder. Therefore Intel's missing GDN logs are likely a logging/control-flow difference, not proof that GDN-like fusion is absent.

Practical reproduction target after this evidence:

1. Use `qwen35moe-no-fa-graph.json` as the main reference, not the FA-enabled graph.
2. Preserve or explicitly fuse `GATED_DELTA_NET` linear-attention cores.
3. Add a non-FA attention pattern pass for `KQ MUL_MAT -> SOFT_MAX -> KQV MUL_MAT`.
4. Add MoE router/expert reduction fusion, because MoE dominates every layer after attention.
5. Track reductions against `3778 -> 966/906`, not `3729 -> 966/906`.
