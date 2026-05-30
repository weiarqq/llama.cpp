# Qwen3.5 / Qwen3.5-MoE SYCL Optimization Todos

This document summarizes optimization points discussed for `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, and the SYCL backend. It is intended as a local reference for future graph optimization work, ordered by recommendation level.

## Reference Baseline

- Current upstream SYCL skips metadata ops at compute time in `ggml/src/ggml-sycl/ggml-sycl.cpp`:
  `RESHAPE`, `TRANSPOSE`, `VIEW`, `PERMUTE`, `NONE`, and empty tensors.
- The scheduler already has a backend graph optimization hook in `ggml/src/ggml-backend.cpp` via `ggml_backend_graph_optimize(...)`.
- Current upstream SYCL does not implement `.graph_optimize`.
- Qwen3.5 linear attention uses `build_layer_attn_linear()` in `src/models/qwen35.cpp`.
- Qwen3.5-MoE uses the same linear-attention/GDN structure in `src/models/qwen35moe.cpp`.
- Existing fused GDN entry point is `ggml_gated_delta_net(...)`, called from `src/models/delta-net-base.cpp`.

## Recommendation Legend

- P0: highly recommended, low semantic risk or foundational for later passes.
- P1: recommended, meaningful expected benefit with manageable complexity.
- P2: useful but more backend-specific or requires stronger validation.
- P3: speculative, only after lower-risk passes are validated.

## Todo List

### P0: Add SYCL Graph Optimize Infrastructure

- [ ] Implement a SYCL `.graph_optimize` callback.
- [ ] Start as analysis-only: log before/after node counts, op histograms, and split boundaries.
- [ ] Keep optimization disabled behind a build/runtime guard until validated.
- [ ] Validate against `graph nodes`, compute buffer sizes, and `graph splits`.

Why: Intel's lower `sched_reserve` node counts imply graph rewriting before reservation, not only runtime skipping.

Primary files:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
- `ggml/src/ggml-backend.cpp`

### P0: Prune Metadata/Layout Nodes From Executable Graph

- [ ] Remove executable `VIEW`, `RESHAPE`, `TRANSPOSE`, `PERMUTE`, and `NONE` nodes from optimized graph views.
- [ ] Preserve tensor metadata and tensor objects so consumers still see correct `data`, `ne`, `nb`, `view_src`, and `view_offs`.
- [ ] Rewire only when semantics are unchanged; do not materialize copies.
- [ ] Treat this as pruning, not kernel fusion.

Why: SYCL already treats these as no-op at execution time. Moving the skip into graph optimization reduces node count and scheduler/planner overhead.

Examples:

- `ggml_view_1d()` creates metadata-only views.
- `s_copy_main = ggml_view_1d(ctx0, s_copy, n_seqs, 0)` is consumed by `GET_ROWS` and does not need a kernel.

### P0: Prune Empty Tensors and Dead Cache Branches

- [ ] Remove nodes where `ggml_is_empty(node)` is true.
- [ ] Focus on recurrent cache paths that produce zero-sized `GET_ROWS` / `CPY` nodes.
- [ ] Preserve side-effecting cache updates when tensor size is non-zero.

Why: Upstream SYCL already skips empty tensors at compute time; pruning earlier contributes directly to lower graph node counts.

### P1: Fuse `RMS_NORM + MUL`

- [ ] Detect `RMS_NORM` followed by weight `MUL` in graph optimization.
- [ ] Fuse where the norm output has a single consumer or where aliasing is proven safe.
- [ ] Use existing CPU/backend fusion patterns as precedent.

Why: Qwen layers repeatedly build norms as `RMS_NORM(input)` followed by multiplication by layer norm weights. This appears as names like `norm-0` and `attn_norm-0`.

Relevant source:

- `src/llama-graph.cpp` `build_norm()`
- `src/models/qwen35.cpp` attention norm and post-attention norm calls
- `src/models/qwen35moe.cpp` equivalent paths

### P1: Fuse `beta + alpha` Input Projections

- [ ] Combine `ssm_beta` and `ssm_alpha` projections into one `MUL_MAT` where model tensors support it.
- [ ] Split the result with metadata views into beta and alpha regions.
- [ ] Preserve per-tensor scale handling: `ssm_beta_s` and `ssm_alpha_s` may differ.
- [ ] Account for LoRA in `build_lora_mm()` before changing graph construction globally.

Why: Both projections consume the same `cur` input and are independent. `qwen3next.cpp` already uses the analogous `ssm_beta_alpha` fused tensor path.

Current Qwen35 source:

```cpp
beta  = build_lora_mm(model.layers[il].ssm_beta,  cur, model.layers[il].ssm_beta_s);
alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
```

Reference pattern:

- `src/models/qwen3next.cpp` uses `ssm_beta_alpha` and then splits with `VIEW`.

### P1: Fuse GDN Core and Immediate Input Preparation

- [ ] Keep current `ggml_gated_delta_net(q, k, v, g, b, s)` as the core fused GDN boundary.
- [ ] Extend backend pattern fusion around the core to include immediate producers where safe:
  `SSM_CONV -> SILU -> q/k/v VIEW split -> L2_NORM(q,k) -> GATED_DELTA_NET`.
- [ ] Avoid forcing materialization for view splits.
- [ ] Validate both autoregressive (`n_seq_tokens == 1`) and chunked paths.

Why: Current fused GDN starts only after `q_conv`, `k_conv`, `v_conv`, `gate`, `beta`, and `state` are prepared. Most remaining linear-attention nodes are in this preparation region.

Current Qwen35 boundary:

```text
conv_input + conv_kernel
  -> SSM_CONV
  -> SILU
  -> split q/k/v
  -> L2_NORM(q,k)
  -> GATED_DELTA_NET(q,k,v,gate,beta,state)
```

Relevant source:

- `src/models/qwen35.cpp` lines around `ggml_ssm_conv`, `ggml_silu`, `q_conv/k_conv/v_conv`, `ggml_l2_norm`, `build_delta_net(...)`
- `src/models/delta-net-base.cpp` `build_delta_net_fused()`

### P1: Fuse GDN Gate/Beta Preprocessing

- [ ] Fuse `beta -> sigmoid`.
- [ ] Fuse `alpha -> add(ssm_dt) -> softplus -> mul(ssm_a)` into gate preprocessing.
- [ ] Consider combining this with the `beta + alpha` projection fusion.
- [ ] Keep output shapes identical to current `beta` and `gate` tensors before feeding GDN.

Why: These are deterministic elementwise chains immediately producing GDN inputs.

Current Qwen35 source:

```text
beta  = W_beta  * cur -> reshape -> sigmoid
alpha = W_alpha * cur -> reshape -> add dt -> softplus -> mul ssm_a -> reshape -> gate
```

### P1: Fuse GDN Output Gating/Normalization

- [ ] Detect `GDN output -> RMS_NORM -> SILU(z) -> MUL`.
- [ ] Fuse the `build_norm_gated()` path where shapes and consumers are simple.
- [ ] Keep `ssm_out` projection separate initially; consider GEMM epilogue fusion later.

Why: `build_norm_gated()` is a compact post-GDN elementwise/norm chain:

```cpp
normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
gated_silu = ggml_silu(ctx0, gate);
return ggml_mul(ctx0, normalized, gated_silu);
```

Relevant source:

- `src/models/qwen35.cpp` `build_norm_gated()`
- `src/models/qwen35moe.cpp` equivalent function

### P2: Fuse `qkv + z` Input Projections

- [ ] Combine `wqkv` and `wqkv_gate` projections when scales/LoRA are compatible or handled per slice.
- [ ] Split the fused output into `qkv_mixed` and `z` with metadata views.
- [ ] Ensure `qkv_mixed` remains efficient for the later transpose/conv path.
- [ ] Avoid adding extra `CONT` nodes that erase the benefit.

Why: Both projections consume the same `cur` input and are independent.

Current Qwen35 source:

```cpp
qkv_mixed = build_lora_mm(model.layers[il].wqkv,      input, model.layers[il].wqkv_s);
z         = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
```

### P2: Fuse All Four Linear-Attention Input Projections

- [ ] Consider one large projection for `qkv_mixed`, `z`, `beta`, and `alpha`.
- [ ] Split the result into four metadata views.
- [ ] Preserve separate per-tensor scales: `wqkv_s`, `wqkv_gate_s`, `ssm_beta_s`, `ssm_alpha_s`.
- [ ] Preserve LoRA behavior from `build_lora_mm()` for each original weight.
- [ ] Validate memory layout for all downstream consumers.

Why: This can reduce four `MUL_MAT` ops to one. It is especially attractive for token generation where small GEMM launch overhead is significant.

Risk: More complex than `beta + alpha` because consumers differ: `qkv_mixed` feeds convolution, `z` feeds gated norm, `beta` feeds sigmoid/GDN, and `alpha` feeds gate construction.

### P2: Optimize Recurrent Cache `VIEW -> GET_ROWS`

- [ ] Do not execute the `VIEW`; keep it as metadata.
- [ ] Let `GET_ROWS` consume the view metadata directly.
- [ ] Do not replace `src[1]` with the full index tensor unless output shape and offset semantics are preserved.

Why: `s_copy_main` is a view over `s_copy`. It tells `GET_ROWS` which recurrent slots to load. The view itself is no-op, but the shape and offset are semantically important.

Example source:

- `src/llama-graph.cpp` `build_rs_inp_impl()` creates `s_copy_main` and `s_copy_extra`.
- `src/llama-graph.cpp` `build_rs()` uses `get_state_rows(ctx0, states, state_copy_main)`.

### P2: Optimize Recurrent State Clear/Copy Paths Carefully

- [ ] Treat `ggml_scale_inplace(state_zero, 0)` as a side-effecting cache clear.
- [ ] Consider replacing it with a backend fill-zero/memset path.
- [ ] Do not fuse it into `GET_ROWS` unless read/write regions and ordering are fully proven.
- [ ] Preserve recurrent state updates into `cache_r_l*` and `cache_s_l*`.

Why: These nodes modify recurrent cache state. They are not ordinary pure elementwise ops.

Relevant source:

- `src/llama-graph.cpp` `build_rs()`
- `src/models/qwen35.cpp` recurrent cache load/update around `conv_states_all` and `ssm_states_all`

### P2: Fuse Non-FA Full Attention Pattern

- [ ] For FA-disabled graphs, detect `KQ MUL_MAT -> scale/mask/softmax -> KQV MUL_MAT`.
- [ ] Replace with a backend attention primitive where possible.
- [ ] Preserve behavior when Flash Attention is explicitly disabled or unsupported.

Why: Intel logs showed Flash Attention disabled while node counts were still much lower. Full attention layers likely use pattern fusion rather than `FLASH_ATTN_EXT`.

### P2: Fuse FFN Patterns

- [ ] Detect dense FFN pattern in Qwen35: `up`, `gate`, activation, multiply, `down`.
- [ ] Preserve LoRA and per-tensor scales.
- [ ] Consider backend-specific fused FFN kernels or graph-level compaction.

Why: Qwen35 uses dense FFN, while Qwen35-MoE uses MoE. FFN contributes recurring per-layer node count outside the GDN path.

Relevant source:

- `src/models/qwen35.cpp` `build_layer_ffn()`
- `src/llama-graph.cpp` `build_ffn()`

### P2: Fuse MoE Router/Expert Paths for Qwen35-MoE

- [ ] For Qwen35-MoE, detect router/top-k/expert dispatch/reduce patterns.
- [ ] Preserve per-expert scales and `build_lora_mm_id()` semantics.
- [ ] Validate output identity for sparse expert routing.

Why: Qwen35-MoE node count reduction cannot be explained only by GDN and metadata pruning. MoE router/expert subgraphs are likely another major target.

Relevant source:

- `src/models/qwen35moe.cpp` `build_layer_ffn()`
- `src/llama-graph.cpp` MoE helpers around expert selection and `build_lora_mm_id()`

### P3: Direct GDN Write-Back to Recurrent State Cache

- [ ] Explore extending fused GDN to write `new_state` directly into the recurrent state target view.
- [ ] Keep graph semantics explicit unless ggml supports multi-output or side-effecting backend ops safely.
- [ ] Validate ordering with downstream consumers and scheduler split boundaries.

Why: Current graph creates fused GDN result, views `new_state`, then copies it into `ssm_states_all`. Direct write-back could reduce nodes and memory traffic, but the side effect makes this higher risk.

Current source:

```cpp
auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);
new_state = attn_out.second;
ggml_cpy(ctx0, new_state, ggml_view_2d(ctx0, ssm_states_all, ...));
```

### P3: Direct Convolution State Cache Update

- [ ] Explore direct update of `conv_states_all` from the convolution input path.
- [ ] Avoid removing the update unless it is proven redundant for the current token/sequence layout.
- [ ] Validate prompt and token-generation paths separately.

Why: `last_conv_states -> CPY -> state_update_target` is side-effecting cache maintenance. It may be optimized, but not treated as dead computation.

Relevant source:

- `src/models/qwen35.cpp` `last_conv_states`, `state_update_target`, and `ggml_cpy(...)`
- `src/models/qwen35moe.cpp` equivalent path

## Suggested Implementation Order

1. Add analysis-only SYCL `graph_optimize` logging.
2. Prune metadata/layout nodes and empty nodes.
3. Add `RMS_NORM + MUL` fusion.
4. Add `beta + alpha` projection fusion, using Qwen3Next as reference.
5. Add GDN input preparation fusion: `SSM_CONV + SILU + q/k/v split + L2_NORM + GATED_DELTA_NET`.
6. Add GDN gate/beta preprocessing fusion.
7. Add GDN output `RMS_NORM + SILU(z) + MUL` fusion.
8. Evaluate `qkv + z` and four-projection fusion.
9. Add full-attention and FFN/MoE pattern fusions.
10. Only then explore direct cache write-back optimizations.

## Validation Checklist

- [ ] Compare `graph nodes` before and after each pass.
- [ ] Compare `SYCL0 compute buffer size` and host compute buffer size.
- [ ] Confirm `graph splits` remains stable unless intentionally changed.
- [ ] Run both prompt-processing and token-generation paths.
- [ ] Validate with Flash Attention enabled and disabled.
- [ ] Validate with fused GDN autoregressive and chunked modes.
- [ ] Validate Qwen35 and Qwen35-MoE separately.
- [ ] Include LoRA/per-tensor-scale cases before enabling projection fusion generally.

## Notes on Risk

- Metadata pruning is low risk only if tensor metadata remains valid for consumers.
- Recurrent cache clear/copy nodes are side-effecting and must not be deleted as ordinary dead code.
- Projection fusion must preserve separate scales and LoRA semantics from `build_lora_mm()`.
- GDN expansion beyond `ggml_gated_delta_net` should be validated independently for AR and chunked paths.
