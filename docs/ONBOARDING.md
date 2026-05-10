# llama.cpp 新手入门指南

> 本文档由知识图谱自动生成，基于 commit `3f8752b5592fdef6fe58fb23a8d468f603095154`
> 生成日期：2026-04-14

## 项目概述

**llama.cpp** 是一个纯 C/C++ 的 LLM（大语言模型）推理引擎，以极少的外部依赖实现了在各种硬件上的高效推理。

- **语言**：C, C++, Python
- **核心框架**：ggml（内嵌 vendored 副本）
- **描述**：支持多硬件后端（CPU、CUDA、HIP、Metal、Vulkan、SYC L 等）、量化格式（1-8bit）、130+ 模型架构
- **主要二进制**：
  - `llama-cli` — 交互式文本生成 / repo-style 推理
  - `llama-server` — OpenAI 兼容的 REST API 服务器

## 架构分层

项目共分为 **9 个架构层**，从顶层到底层：

### Layer 1 — Public API（公共 API）

**描述**：稳定的 C/C++ API 面，向应用程序暴露，是应用程序唯一直接链接的层。

| 文件 | 用途 |
|------|------|
| `include/llama.h` | 主公共 C API 头文件，定义 `llama_model`、`llama_context`、`llama_sampler`、`llama_vocab`、`llama_batch` 等结构 |
| `include/llama-cpp.h` | C++ RAII 包装（`std::unique_ptr` 封装） |

### Layer 2 — Core Library（核心库）

**描述**：llama.cpp 的内部实现，包括模型加载、上下文管理、KV-cache、图构建、分词器、采样器、批处理、内存抽象、量化、适配器等。所有计算图在此组装后委托给 GGML。

| 文件 | 用途 |
|------|------|
| `src/llama.cpp` | C API 入口实现（`llama_model_load_from_file`、`llama_init_from_model` 等） |
| `src/llama-arch.cpp` | 所有支持的模型架构的名称映射、KV 键名构建器、张量信息注册 |
| `src/llama-arch.h` | `llm_arch`（架构枚举）、`llm_kv`（KV 元数据键枚举）、`llm_tensor`（张量角色枚举）、`LLM_KV`/`LLM_TN` 辅助类 |
| `src/llama-model.cpp` | **核心**：GGUF 解析、架构检测、张量加载、权重分配，包含所有 130+ 架构的图构建器（9649 行） |
| `src/llama-model-loader.cpp` | GGUF 文件读取、张量创建、数据加载 |
| `src/llama-mmap.cpp` | 内存映射文件读取，支持大模型分片加载 |
| `src/llama-vocab.cpp` | 三种分词器实现：SentencePiece (SPM)、BPE、WPM；50+ 模型特定预分词正则模式 |
| `src/llama-sampler.cpp` | 15+ 采样策略：greedy、multinomial、temperature、top-k、top-p、min-p、重复惩罚、语法约束等 |
| `src/llama-kv-cache.cpp` | KV-cache 管理：槽位分配、缓存驱逐、滑动窗口 / ISWA 支持 |
| `src/llama-batch.cpp` | 批次拆分与微批次分配 |
| `src/llama-graph.cpp` | 计算图组装（llm_graph_context），将每一层（attention、FFN、MoE、RoPE 等）编织为 GGML 调用序列（2830 行） |
| `src/llama-memory.cpp` | 内存抽象接口 |
| `src/llama-memory-hybrid*.cpp` | 混合内存实现（递归 + 滑动窗口注意力） |
| `src/llama-memory-recurrent.cpp` | 递归内存（Mamba/RWKV 状态管理） |
| `src/llama-io.h/cpp` | 抽象 I/O 接口（读写数据和张量） |
| `src/llama-cparams.h/cpp` | 内部推理参数（RoPE 缩放、YaRN 超参等） |

### Layer 3 — Model Implementations（模型实现）

**描述**：130+ 种具体模型架构的 GGML 图构建器，每个文件定义一个 `llm_build_*` 类来构建该架构的 GGML 计算图。

**主要架构分组**：

| 架构族 | 代表文件 |
|--------|----------|
| LLaMA 系列 | `src/models/llama.cpp`, `src/models/llama-iswa.cpp` |
| Qwen 系列 | `src/models/qwen.cpp`, `qwen2*.cpp`, `qwen3*.cpp`, `qwen3vl*.cpp`, `qwen3moe*.cpp` |
| DeepSeek 系列 | `src/models/deepseek.cpp`, `src/models/deepseek2.cpp` |
| Gemma 系列 | `src/models/gemma.cpp`, `gemma2-iswa.cpp`, `gemma3*.cpp`, `gemma-embedding.cpp` |
| Mistral 系列 | `src/models/mistral3.cpp` |
| MoE 模型 | `src/models/qwen2moe.cpp`, `src/models/deepseek2.cpp`, `src/models/afmoe.cpp`, `src/models/bailingmoe*.cpp` |
| 状态空间模型 | `src/models/mamba.cpp`, `src/models/rwkv6*.cpp`, `src/models/rwkv7*.cpp` |
| BERT 系列 | `src/models/bert.cpp`, `src/models/neo-bert.cpp`, `src/models/modern-bert.cpp` |
| 多模态 | `src/models/qwen2vl.cpp`, `src/models/chatglm.cpp`, `src/models/cogvlm.cpp`, `src/models/chameleon.cpp` |
| 扩散模型 | `src/models/llada.cpp`, `src/models/llada-moe.cpp`, `src/models/dream.cpp` |

### Layer 4 — ggml Tensor Library（GGML 张量库）

**描述**：GGML 的核心——定义 `ggml_tensor`、`ggml_context`、`ggml_cgraph`、96 种张量操作、图分配器、GGUF 文件格式、后端接口。这是所有模型计算的基础。

| 文件 | 用途 |
|------|------|
| `ggml/include/ggml.h` | **核心头文件**：96 种张量操作、42 种数据类型、20+ 量化格式、`ggml_tensor`/`ggml_cgraph`/`ggml_backend` API |
| `ggml/include/ggml-cpp.h` | C++ RAII 包装 |
| `ggml/include/ggml-alloc.h` | 图分配器（支持 inplace、单节点复用） |
| `ggml/include/ggml-backend.h` | 硬件抽象层：CPU/CUDA/Metal/Vulkan 等后端接口、调度器、多设备张量并行 |
| `ggml/include/gguf.h` | GGUF 文件格式读/写 |
| `ggml/src/ggml.c` | **纯 C 张量操作层**：声明式图构建（所有 `ggml_*()` 工厂函数），无实际计算（7770 行） |
| `ggml/src/ggml.cpp` | **C++ 计算层**：`ggml_compute_forward_*()` 实际执行张量操作，SIMD 循环，调用后端 |
| `ggml/src/ggml-alloc.c` | 图分配器实现，支持即时 inplace 操作 |
| `ggml/src/ggml-quants.c` | 所有量化格式的实现（5491 行） |
| `ggml/src/ggml-backend.cpp` | 后端注册与管理 |
| `ggml/src/gguf.cpp` | GGUF 格式读/写实现 |

> **关键理解**：`ggml.c` 是纯声明式的（只构建计算图，不做计算），`ggml.cpp` 是实际执行层（SIMD 循环、后端分发）。两者配合工作。

### Layer 5 — ggml CPU Backend（CPU 后端）

**描述**：在 x86（AVX/AVX2/AVX512）、ARM NEON、PPC VSX、RISC-V RVV、IBM s390x VXE、LoongArch LSX、WebAssembly SIMD128 上实现所有张量操作的 SIMD 向量化。包含量化矩阵乘法内核、权重重打包、BLAS 集成、线程池和 NUMA 支持。

| 文件 | 用途 |
|------|------|
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU 后端入口、调度 |
| `ggml/src/ggml-cpu/quants.c` | 量化内核实现 |
| `ggml/src/ggml-cpu/vec.cpp` | SIMD 向量操作（所有 arch 共用） |
| `ggml/src/ggml-cpu/simd-gemm.h` | 通用 GEMM 模板 |
| `ggml/src/ggml-cpu/simd-mappings.h` | SIMD 后端选择宏 |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | x86 SIMD（AVX/AVX2/AVX512）量化 |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | ARM NEON 量化 |
| `ggml/src/ggml-cpu/arch/powerpc/quants.c` | PPC VSX 量化 |
| `ggml/src/ggml-cpu/arch/riscv/quants.c` | RISC-V RVV 量化 |
| `ggml/src/ggml-cpu/arch/s390/quants.c` | IBM s390x VXE/NNPA 量化 |
| `ggml/src/ggml-cpu/arch/loongarch/quants.c` | LoongArch LSX 量化 |
| `ggml/src/ggml-cpu/arch/wasm/quants.c` | WebAssembly SIMD128 量化 |
| `ggml/src/ggml-cpu/amx/` | Intel AMX 加速 |
| `ggml/src/ggml-cpu/kleidiai/` | KleidiAI 加速 |
| `ggml/src/ggml-cpu/llamafile/` | Llamafile 加速 |
| `ggml/src/ggml-cpu/spacemit/` | Spacemit 加速 |

### Layer 6 — ggml GPU Backends（GPU 后端）

**描述**：14 种 GPU 加速后端，每个实现 `ggml_backend_i` 接口。

| 后端 | 核心文件 |
|------|----------|
| NVIDIA CUDA | `ggml/src/ggml-cuda/ggml-cuda.cu`（5315 行）、`mmul.cu`、`fattn.cu`、`mmq.cu` |
| Apple Metal | `ggml/src/ggml-metal/ggml-metal.metal`（10549 行 shader）|
| Vulkan | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` |
| Intel SYCL | `ggml/src/ggml-sycl/ggml-sycl.cpp` |
| Huawei CANN | `ggml/src/ggml-cann/ggml-cann.cpp` |
| Qualcomm Hexagon | `ggml/src/ggml-hexagon/ggml-hexagon.cpp` |
| BLAS | `ggml/src/ggml-blas/ggml-blas.cpp` |
| WebGPU | `ggml/src/ggml-webgpu/ggml-webgpu.cpp` |
| OpenCL | `ggml/src/ggml-opencl/ggml-opencl.cpp` |
| RPC | `ggml/src/ggml-rpc/ggml-rpc.cpp` |

### Layer 7 — Common Utilities（通用工具）

**描述**：CLI 工具和共享工具，供 `llama-cli`、`llama-server` 使用。

| 文件 | 用途 |
|------|------|
| `common/arg.cpp` | 命令行参数解析 |
| `common/console.cpp` | 终端 I/O（颜色、进度条） |
| `common/log.cpp` | 异步日志系统 |
| `common/download.cpp` | 文件下载（HuggingFace、URL、Docker） |
| `common/hf-cache.cpp` | HuggingFace 缓存管理 |
| `common/chat.cpp` | 聊天会话处理、Jinja2 模板（2190 行） |
| `common/sampling.cpp` | 采样包装器 |
| `common/chat-peg-parser.cpp` | PEG 解析器 |
| `common/json-schema-to-grammar.cpp` | JSON Schema → GBNF 语法转换 |
| `common/speculative.cpp` | 投机解码（n-gram、自投机） |
| `common/reasoning-budget.cpp` | 思考预算管理 |
| `common/jinja/` | Jinja2 模板引擎（词法分析器/解析器/运行时） |
| `common/unicode.cpp` | Unicode 工具 |
| `common/debug.cpp` | 调试工具 |

### Layer 8 — Python Tools（Python 工具）

**描述**：模型转换脚本和 GGUF Python 包。

| 文件 | 用途 |
|------|------|
| `convert_hf_to_gguf.py` | HuggingFace → GGUF 转换（13236 行，最长文件） |
| `convert_lora_to_gguf.py` | LoRA 适配器转换 |
| `convert_llama_ggml_to_gguf.py` | 旧版 GGML → GGUF 迁移 |
| `gguf-py/gguf/gguf.py` | GGUF 读写核心 |
| `gguf-py/gguf/gguf_reader.py` | GGUF 文件读取 |
| `gguf-py/gguf/gguf_writer.py` | GGUF 文件写入 |
| `gguf-py/gguf/vocab.py` | 分词器处理 |
| `gguf-py/gguf/quants.py` | 量化逻辑 |

### Layer 9 — Build and Infrastructure（构建与基础设施）

**描述**：CMake 构建配置、CI/CD 工作流、Docker 镜像、Nix 打包。

| 文件 | 用途 |
|------|------|
| `CMakeLists.txt` | 主构建文件 |
| `CMakePresets.json` | 构建预设 |
| `ggml/CMakeLists.txt` | ggml 子项目构建 |
| `.github/workflows/` | CI/CD 工作流（server、docker、build、cann、vulkan） |
| `.devops/*.Dockerfile` | 多架构 Docker 镜像（CPU、CUDA、ROCm、Vulkan） |
| `docs/build.md` | 构建指南 |
| `docs/development/HOWTO-add-model.md` | 添加新模型指南 |

---

## 核心概念

### 计算图范式

llama.cpp 的推理核心是**声明式计算图**：
1. 使用 `ggml_*()` 工厂函数构建张量操作（不执行计算）
2. 调用 `ggml_build_forward_expand()` 固化图
3. 调用 `ggml_graph_compute()` 执行

这使得计算可以跨硬件后端分发，且支持图融合优化。

### GGUF 文件格式

所有模型以 GGUF 格式分发，包含：
- KV 元数据（超参数、分词器配置、RoPE 参数）
- 分词器词汇
- 张量数据（权重，支持多种量化格式）

### 量化系统

支持 20+ 量化格式：Q1-Q8（普通 + K 变体）、IQ 系列（int4）、MXFP4、NVFP4、TQ1/TQ2 等。量化直接影响推理速度与精度权衡。

### 架构注册机制

新增模型需要修改：
1. `src/llama-arch.h/.cpp` — 注册架构名、KV 键、张量角色
2. `convert_hf_to_gguf.py` — 添加模型映射子类
3. `src/llama-model.cpp` — 实现图构建器

### 内存管理

- `llama_mmap` — 内存映射文件，支持部分加载（GPU 分层卸载）
- `llama_kv_cache` — KV 存储，支持 ISWA（交错滑动窗口注意力）和递归缓存
- 多 GPU — `llama_params_fit` 自动将层分布到多卡

### 后端调度

`ggml_backend_scheduler` 支持将计算图分区到多个设备（多 GPU 张量并行），通过事件协调。

---

## 引导学习路径（15 步）

### 第 1 步：项目概览
阅读 `README.md`、`include/llama.h`、`ggml/include/ggml.h`。理解三层架构：ggml（底层张量计算）→ src/（LLM 前向传播）→ common/（工具）。

### 第 2 步：公共 C API
`include/llama.h` 定义 6 种核心类型。标准推理流程：
```
llama_backend_init()
→ llama_model_load_from_file()
→ llama_init_from_model()
→ loop: tokenize → llama_decode → sample → ...
→ llama_model_free + llama_backend_free
```

### 第 3 步：模型加载
`src/llama-model.cpp`（9649 行）负责 GGUF 解析、架构检测、张量加载，处理 Q4_K_M、IQ4_XS、MXFP4 等量化格式。

### 第 4 步：架构注册
`src/llama-arch.h/.cpp` 是添加新模型的入口点。先修改这里，再动其他代码。

### 第 5 步：分词器
`src/llama-vocab.cpp` 实现 SPM/BPE/WPM 三种分词算法，处理 50+ 模型特定的预分词正则模式。

### 第 6 步：KV Cache 和批处理
`src/llama-kv-cache.cpp` 管理缓存槽位分配与驱逐；`src/llama-batch.cpp` 拆分微批次。

### 第 7 步：计算图构建
`src/llama-graph.cpp`（2830 行）将每一层编织为 GGML 调用序列，是模型推理的核心编排逻辑。

### 第 8 步：GGML 张量库
`ggml/include/ggml.h` 是整个项目的基础。核心模式：创建张量 → 固化图 → 执行。

### 第 9 步：后端抽象
`ggml/include/ggml-backend.h` 提供硬件抽象，调度器可跨多 GPU 分发计算。

### 第 10 步：CUDA 后端
`ggml/src/ggml-cuda/ggml-cuda.cu`（5315 行）使用 cuBLAS 做 GEMM，自定义 kernel 做 attention/norm/RoPE。

### 第 11 步：Metal 后端
`ggml/src/ggml-metal/ggml-metal.metal`（10549 行 shader）是 Apple Silicon 的一级公民。

### 第 12 步：Vulkan 后端
`ggml/src/ggml-vulkan/ggml-vulkan.cpp` 是跨厂商 GPU 路径（AMD/Intel/Qualcomm）。

### 第 13 步：Token 采样
`src/llama-sampler.cpp` 实现 15+ 采样策略，可链式组合。

### 第 14 步：Chat 模板和自动解析
`common/chat.cpp` 用 Jinja2 + PEG 解析器处理多轮对话；`common/json-schema-to-grammar.cpp` 保证 JSON 输出合法。

### 第 15 步：模型转换
`convert_hf_to_gguf.py`（13236 行）将 HuggingFace 模型转为 GGUF，是添加新模型的最后一站。

---

## 复杂度热点

以下文件规模大或逻辑复杂，接触时需格外小心：

| 文件 | 行数 | 原因 |
|------|------|------|
| `ggml-metal/ggml-metal.metal` | 10549 | Metal shader 代码量大，kernel 众多 |
| `convert_hf_to_gguf.py` | 13236 | 转换逻辑复杂，需要处理各种模型格式差异 |
| `ggml-cuda/ggml-cuda.cu` | 5315 | CUDA kernel 注册表，架构复杂 |
| `ggml/src/ggml-quants.c` | 5491 | 所有量化算法实现，数学密集 |
| `src/llama-model.cpp` | 9649 | 所有 130+ 架构的图构建器集中于此 |
| `ggml/src/ggml.c` | 7748 | 96 种张量操作 + 完整反向传播，代码量大 |
| `src/llama-graph.cpp` | 2830 | 图组装逻辑复杂，涉及大量张量 reshape/permute |
| `common/chat.cpp` | 2190 | Jinja2 + 多轮对话 + 工具调用逻辑 |

---

## 快速开始

### 构建
```bash
cmake -B build
cmake --build build --config Release -j 8
# 测试
ctest --test-dir build
```

### 添加新模型
1. 在 `src/llama-arch.h/.cpp` 注册架构
2. 在 `convert_hf_to_gguf.py` 添加 `ModelBase` 子类
3. 在 `src/llama-model.cpp` 实现 `llm_build_*()` 函数
4. 运行 `ctest --test-dir build` 验证

### 修改 GGML
如果修改了 `ggml/` 源代码，运行 `test-backend-ops` 验证后端一致性：
```bash
ctest --test-dir build -R test-backend-ops
```

---

## 更多资源

- [构建指南](docs/build.md)
- [添加模型指南](docs/development/HOWTO-add-model.md)
- [服务器使用](tools/server/README.md)
- [服务器开发](tools/server/README-dev.md)
- [调试测试](docs/development/debugging-tests.md)
