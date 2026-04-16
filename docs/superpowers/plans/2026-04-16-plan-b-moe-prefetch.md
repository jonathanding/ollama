# Plan B MoE Expert 预取流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `ggml_backend_sched_compute_splits` 中实现 MoE expert 全量预取，使 N+1 层的 Host-to-Device (H2D) copy 与第 N 层的 GPU compute 并行执行，验证 prefill 是否低于 1360 ms。

**Architecture:** `ggml-backend.cpp` 不含 CUDA headers，通过 proc address 机制（已有先例：`ggml_backend_register_host_buffer`）从 `ggml-cuda.cu` 暴露 8 个 CUDA stream/event/memcpy 薄包装函数，`ggml-backend.cpp` 用 `void*` 不透明句柄调用这些函数。预取逻辑完全封装在 `ggml_backend_sched_compute_splits` 局部变量中，不修改 `ggml_backend_sched` 结构体。

**Tech Stack:** Go 1.24, CGo, C++17 (ggml-backend.cpp), CUDA C++ (ggml-cuda.cu), Windows 11 + CUDA 13, PowerShell build

---

## 文件改动清单

| 文件 | 类型 | 职责 |
|---|---|---|
| `envconfig/config.go` | 修改 | 新增 `MoePrefetch` env var |
| `llm/server.go` | 修改 | `MoePrefetch` 为 true 时注入 `OLLAMA_MOE_PREFETCH=1` 到 runner 子进程 |
| `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` | 修改 | 实现 8 个 CUDA 薄包装函数，注册为 proc addresses |
| `ml/backend/ggml/ggml/src/ggml-backend.cpp` | 修改 | `ggml_backend_sched_compute_splits` 内：辅助函数 + 预取主循环 |

---

## Task 1: Go 层 — 新增 `OLLAMA_MOE_PREFETCH` env var

**Files:**
- Modify: `envconfig/config.go:232` (紧跟 `MoePinned` 之后)
- Modify: `llm/server.go:271-276` (紧跟 `MoePinned` 注入块之后)

- [ ] **Step 1: 在 `envconfig/config.go` 添加 `MoePrefetch`**

在 `MoePinned = Bool("OLLAMA_MOE_PINNED")` 之后，`NoHistory` 之前，插入：

```go
// MoePrefetch enables lookahead copy inside ggml_backend_sched_compute_splits:
// after submitting GPU compute for split N, immediately fires an async H2D copy
// of split N+1's MoE expert weights on an independent CUDA copy stream.
// Requires OLLAMA_MOE_PINNED=1 (pinned source memory for true copy/compute overlap).
// Default: false.
MoePrefetch = Bool("OLLAMA_MOE_PREFETCH")
```

- [ ] **Step 2: 在 `llm/server.go` 注入 `OLLAMA_MOE_PREFETCH` 到子进程 env**

在现有的 `if envconfig.MoePinned() && envconfig.MoeGpuLayers() != 0 { ... }` 块之后（约第 276 行），插入：

```go
if envconfig.MoePrefetch() && envconfig.MoePinned() && envconfig.MoeGpuLayers() != 0 {
    if runnerEnvs == nil {
        runnerEnvs = map[string]string{}
    }
    runnerEnvs["OLLAMA_MOE_PREFETCH"] = "1"
}
```

- [ ] **Step 3: 构建 Go 层，确认编译通过**

```
go build ./envconfig/... ./llm/...
```

Expected: 无错误，无输出。

- [ ] **Step 4: Commit**

```
git add envconfig/config.go llm/server.go
git commit -m "feat: add OLLAMA_MOE_PREFETCH env var and runner injection"
```

---

## Task 2: ggml-cuda.cu — 实现 CUDA 薄包装函数 + 注册 proc addresses

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:5076-5091` (proc address 注册表) + 新增函数（在注册表之前）

这些函数使用 `void*` 作为不透明句柄，使 `ggml-backend.cpp` 不需要包含 CUDA headers。

- [ ] **Step 1: 在 `ggml-cuda.cu:5076`（`ggml_backend_cuda_reg_get_proc_address` 函数）之前插入 8 个薄包装函数**

找到 `static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {`（约第 5076 行），在其**之前**插入：

```cpp
// ---------------------------------------------------------------------------
// Plan B MoE prefetch CUDA helpers — exposed via proc address as void* handles
// ---------------------------------------------------------------------------

static void * ggml_backend_cuda_moe_stream_create() {
    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        return nullptr;
    }
    return (void *)stream;
}

static void ggml_backend_cuda_moe_stream_destroy(void * stream_handle) {
    if (stream_handle) {
        cudaStreamDestroy((cudaStream_t)stream_handle);
    }
}

static void ggml_backend_cuda_moe_stream_synchronize(void * stream_handle) {
    if (stream_handle) {
        cudaStreamSynchronize((cudaStream_t)stream_handle);
    }
}

static void * ggml_backend_cuda_moe_event_create() {
    cudaEvent_t event;
    // cudaEventDisableTiming: 不记录时间戳，降低 overhead
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        return nullptr;
    }
    return (void *)event;
}

static void ggml_backend_cuda_moe_event_destroy(void * event_handle) {
    if (event_handle) {
        cudaEventDestroy((cudaEvent_t)event_handle);
    }
}

static void ggml_backend_cuda_moe_event_record(void * event_handle, void * stream_handle) {
    if (event_handle && stream_handle) {
        cudaEventRecord((cudaEvent_t)event_handle, (cudaStream_t)stream_handle);
    }
}

static void ggml_backend_cuda_moe_event_synchronize(void * event_handle) {
    if (event_handle) {
        cudaEventSynchronize((cudaEvent_t)event_handle);
    }
}

// Full-layer prefetch: copies the entire weight tensor (all experts) from CPU
// pinned memory to the VRAM input_cpy tensor using the independent copy stream.
// Returns true on success. Caller must ensure source is pinned (OLLAMA_MOE_PINNED=1).
static bool ggml_backend_cuda_moe_prefetch_tensor(
        void * stream_handle,
        struct ggml_tensor * input,      // CPU-side weight tensor (source)
        struct ggml_tensor * input_cpy)  // VRAM tensor (destination)
{
    if (!stream_handle || !input || !input_cpy) return false;
    cudaStream_t stream = (cudaStream_t)stream_handle;
    size_t nbytes = ggml_nbytes(input);
    return cudaMemcpyAsync(input_cpy->data, input->data, nbytes,
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
}
```

- [ ] **Step 2: 在 `ggml_backend_cuda_reg_get_proc_address` 函数体内注册 8 个 proc addresses**

在现有的 `if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) { ... }` 块之后，`return nullptr;` 之前，插入：

```cpp
    if (strcmp(name, "ggml_backend_cuda_moe_stream_create") == 0) {
        return (void *)ggml_backend_cuda_moe_stream_create;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_stream_destroy") == 0) {
        return (void *)ggml_backend_cuda_moe_stream_destroy;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_stream_synchronize") == 0) {
        return (void *)ggml_backend_cuda_moe_stream_synchronize;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_event_create") == 0) {
        return (void *)ggml_backend_cuda_moe_event_create;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_event_destroy") == 0) {
        return (void *)ggml_backend_cuda_moe_event_destroy;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_event_record") == 0) {
        return (void *)ggml_backend_cuda_moe_event_record;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_event_synchronize") == 0) {
        return (void *)ggml_backend_cuda_moe_event_synchronize;
    }
    if (strcmp(name, "ggml_backend_cuda_moe_prefetch_tensor") == 0) {
        return (void *)ggml_backend_cuda_moe_prefetch_tensor;
    }
```

- [ ] **Step 3: 构建 CUDA backend，确认编译通过**

```powershell
# 仅编译 ggml-cuda target（增量构建，不 clean）
cmake --build build\cuda_v13 --target ggml-cuda --config Release --parallel 8
```

Expected: 无编译错误（`ggml-cuda.dll` 生成成功）。若 build 目录不存在，先运行 `scripts/rebuild_windows.ps1`。

- [ ] **Step 4: Commit**

```
git add ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu
git commit -m "feat: expose CUDA stream/event/memcpy helpers via proc address for MoE prefetch"
```

---

## Task 3: ggml-backend.cpp — 辅助函数和函数指针类型定义

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-backend.cpp` — 在 `ggml_backend_sched_compute_splits`（第 1480 行）之前插入

- [ ] **Step 1: 在 `static enum ggml_status ggml_backend_sched_compute_splits` 之前插入函数指针类型 + `is_moe_cpu_split` 辅助函数**

找到第 1480 行 `static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {`，在其**正上方**插入：

```cpp
// ---------------------------------------------------------------------------
// Plan B MoE prefetch: function pointer types for CUDA helpers via proc address
// ---------------------------------------------------------------------------

typedef void *(*moe_stream_create_fn_t)();
typedef void  (*moe_stream_destroy_fn_t)(void *);
typedef void  (*moe_stream_synchronize_fn_t)(void *);
typedef void *(*moe_event_create_fn_t)();
typedef void  (*moe_event_destroy_fn_t)(void *);
typedef void  (*moe_event_record_fn_t)(void *, void *);
typedef void  (*moe_event_synchronize_fn_t)(void *);
typedef bool  (*moe_prefetch_tensor_fn_t)(void *, struct ggml_tensor *, struct ggml_tensor *);

// Looks up a MoE prefetch proc address from the first GPU backend reg.
// Returns nullptr if no GPU backend or the proc is not found.
static void * moe_get_proc(const char * name) {
    for (int i = 0; i < (int)ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        void * fn = ggml_backend_reg_get_proc_address(reg, name);
        if (fn) return fn;
    }
    return nullptr;
}

// Returns true if split_id refers to a CPU-side MoE expert weight split.
// Detection criterion: the split's first node is GGML_OP_MUL_MAT_ID and its
// weight input is a host (CPU) buffer marked GGML_BACKEND_BUFFER_USAGE_WEIGHTS.
// This is identical to the condition used by ggml-backend.cpp:1517-1521 for
// expert-granular copy, so no new assumptions are introduced.
static bool is_moe_cpu_split(ggml_backend_sched_t sched, int split_id) {
    if (split_id < 0 || split_id >= sched->n_splits) return false;
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    if (split->n_inputs == 0 || split->graph.n_nodes == 0) return false;

    struct ggml_tensor * node = split->graph.nodes[0];
    if (node->op != GGML_OP_MUL_MAT_ID) return false;

    for (int i = 0; i < split->n_inputs; i++) {
        struct ggml_tensor * input = split->inputs[i];
        struct ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
        if (node->src[0] == input_cpy &&
            input->buffer != NULL &&
            ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(input->buffer)) {
            return true;
        }
    }
    return false;
}
```

- [ ] **Step 2: 验证编译通过（CPU-only 编译，不需要 CUDA）**

```powershell
# 仅检查 C++ 编译，不链接
cmake --build build\cpu --target ggml-cpu --config Release --parallel 8
```

Expected: 无编译错误。

- [ ] **Step 3: Commit**

```
git add ml/backend/ggml/ggml/src/ggml-backend.cpp
git commit -m "feat: add is_moe_cpu_split helper and proc address typedefs for Plan B"
```

---

## Task 4: ggml-backend.cpp — 预取状态初始化与清理

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-backend.cpp:1480-1487`（函数开头）和 `1661-1664`（`return GGML_STATUS_SUCCESS` 之前）

- [ ] **Step 1: 在 `ggml_backend_sched_compute_splits` 函数体开头（第 1483 行，`struct ggml_backend_sched_split * splits = sched->splits;` 之后）插入预取初始化块**

找到：
```cpp
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;
```

替换为：
```cpp
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    // Plan B MoE prefetch: initialize stream/event if OLLAMA_MOE_PREFETCH is set
    bool     prefetch_enabled  = false;
    void *   prefetch_stream   = nullptr;
    void *   prefetch_event    = nullptr;
    bool     prefetch_pending  = false;
    int      prefetch_split_id = -1;

    if (getenv("OLLAMA_MOE_PREFETCH") != nullptr) {
        auto fn_stream_create = (moe_stream_create_fn_t)moe_get_proc("ggml_backend_cuda_moe_stream_create");
        auto fn_event_create  = (moe_event_create_fn_t) moe_get_proc("ggml_backend_cuda_moe_event_create");
        if (fn_stream_create && fn_event_create) {
            prefetch_stream = fn_stream_create();
            prefetch_event  = fn_event_create();
            if (prefetch_stream && prefetch_event) {
                prefetch_enabled = true;
                GGML_LOG_DEBUG("%s: MoE prefetch enabled (stream=%p event=%p)\n",
                               __func__, prefetch_stream, prefetch_event);
            } else {
                // Partial failure: clean up and fall back silently
                auto fn_stream_destroy = (moe_stream_destroy_fn_t)moe_get_proc("ggml_backend_cuda_moe_stream_destroy");
                auto fn_event_destroy  = (moe_event_destroy_fn_t) moe_get_proc("ggml_backend_cuda_moe_event_destroy");
                if (fn_stream_destroy && prefetch_stream) fn_stream_destroy(prefetch_stream);
                if (fn_event_destroy  && prefetch_event)  fn_event_destroy(prefetch_event);
                prefetch_stream = nullptr;
                prefetch_event  = nullptr;
            }
        }
    }
```

- [ ] **Step 2: 在 `return GGML_STATUS_SUCCESS;`（约第 1663 行）之前插入清理块**

找到：
```cpp
    return GGML_STATUS_SUCCESS;
}
```

替换为：
```cpp
    // Plan B cleanup: drain the prefetch stream and destroy resources
    if (prefetch_enabled) {
        if (prefetch_pending) {
            auto fn_event_sync = (moe_event_synchronize_fn_t)moe_get_proc("ggml_backend_cuda_moe_event_synchronize");
            if (fn_event_sync) fn_event_sync(prefetch_event);
        }
        auto fn_stream_destroy = (moe_stream_destroy_fn_t)moe_get_proc("ggml_backend_cuda_moe_stream_destroy");
        auto fn_event_destroy  = (moe_event_destroy_fn_t) moe_get_proc("ggml_backend_cuda_moe_event_destroy");
        if (fn_stream_destroy) fn_stream_destroy(prefetch_stream);
        if (fn_event_destroy)  fn_event_destroy(prefetch_event);
    }

    return GGML_STATUS_SUCCESS;
}
```

- [ ] **Step 3: 同样在 compute 失败的 early-return 路径（约第 1618-1620 行）插入清理**

找到：
```cpp
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph, sched->batch_size);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
```

替换为：
```cpp
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph, sched->batch_size);
            if (ec != GGML_STATUS_SUCCESS) {
                if (prefetch_enabled) {
                    auto fn_stream_sync    = (moe_stream_synchronize_fn_t)moe_get_proc("ggml_backend_cuda_moe_stream_synchronize");
                    auto fn_stream_destroy = (moe_stream_destroy_fn_t)    moe_get_proc("ggml_backend_cuda_moe_stream_destroy");
                    auto fn_event_destroy  = (moe_event_destroy_fn_t)     moe_get_proc("ggml_backend_cuda_moe_event_destroy");
                    if (fn_stream_sync)    fn_stream_sync(prefetch_stream);
                    if (fn_stream_destroy) fn_stream_destroy(prefetch_stream);
                    if (fn_event_destroy)  fn_event_destroy(prefetch_event);
                }
                return ec;
            }
```

- [ ] **Step 4: 编译验证**

```powershell
cmake --build build\cuda_v13 --target ggml-cuda --config Release --parallel 8
```

Expected: 无编译错误。

- [ ] **Step 5: Commit**

```
git add ml/backend/ggml/ggml/src/ggml-backend.cpp
git commit -m "feat: add MoE prefetch stream/event init and cleanup in compute_splits"
```

---

## Task 5: ggml-backend.cpp — 主循环预取逻辑

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-backend.cpp` — split 主循环内，两处修改

这是最核心的改动，分两部分：
1. 在 input copy 循环里，对已预取的 MoE weight input 跳过 copy
2. 在 compute 提交后，立刻预取下一层

- [ ] **Step 1: 在 split 主循环的 input copy 部分，添加预取命中检测**

在 `ggml_backend_sched_compute_splits` 的主循环开头（约第 1488 行），找到：
```cpp
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
```

替换为：
```cpp
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // Plan B: if this split was prefetched, wait for the copy to complete
        bool moe_prefetch_hit = false;
        if (prefetch_enabled && prefetch_pending && split_id == prefetch_split_id) {
            auto fn_event_sync = (moe_event_synchronize_fn_t)moe_get_proc("ggml_backend_cuda_moe_event_synchronize");
            if (fn_event_sync) fn_event_sync(prefetch_event);
            prefetch_pending  = false;
            moe_prefetch_hit  = true;
            GGML_LOG_DEBUG("%s: prefetch hit for split %d\n", __func__, split_id);
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
```

- [ ] **Step 2: 在 MoE expert copy 路径中，检查 `moe_prefetch_hit` 并跳过 copy**

在 input copy 内层循环中找到（约第 1517-1599 行）the MoE copy block：
```cpp
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);
```

替换为：
```cpp
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    )) {

                    // Plan B: full-layer prefetch already wrote to input_cpy — skip copy
                    if (moe_prefetch_hit) {
                        GGML_LOG_DEBUG("%s: skipping expert copy for split %d (prefetched)\n",
                                       __func__, split_id);
                        continue;
                    }

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);
```

- [ ] **Step 3: 在 compute 提交后（`ggml_backend_graph_compute_async` 成功返回后），插入预取下一层的逻辑**

找到 compute 提交成功后的位置（约第 1617-1620 行）：
```cpp
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph, sched->batch_size);
            if (ec != GGML_STATUS_SUCCESS) {
                // ... cleanup + return ec (已在 Task 4 插入)
            }
        }
        // callback_eval 路径 ...
```

在这个 `if (!sched->callback_eval) { ... }` 块的**闭合大括号之后**，`// record the event of this copy` 注释之前，插入：

```cpp
        // Plan B: after submitting compute N, fire prefetch for split N+1
        if (prefetch_enabled && !sched->callback_eval && !prefetch_pending) {
            int next_id = split_id + 1;
            if (is_moe_cpu_split(sched, next_id)) {
                struct ggml_backend_sched_split * next_split = &sched->splits[next_id];
                struct ggml_tensor * next_node = next_split->graph.nodes[0];
                auto fn_prefetch = (moe_prefetch_tensor_fn_t)moe_get_proc("ggml_backend_cuda_moe_prefetch_tensor");
                auto fn_event_record = (moe_event_record_fn_t)moe_get_proc("ggml_backend_cuda_moe_event_record");
                if (fn_prefetch && fn_event_record) {
                    bool fired = false;
                    for (int i = 0; i < next_split->n_inputs; i++) {
                        struct ggml_tensor * inp = next_split->inputs[i];
                        struct ggml_tensor * inp_cpy = tensor_copy(inp, next_split->backend_id, sched->cur_copy);
                        if (next_node->src[0] == inp_cpy &&
                            inp->buffer != NULL &&
                            ggml_backend_buffer_get_usage(inp->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                            ggml_backend_buffer_is_host(inp->buffer) &&
                            next_node->op == GGML_OP_MUL_MAT_ID) {
                            if (fn_prefetch(prefetch_stream, inp, inp_cpy)) {
                                fired = true;
                                GGML_LOG_DEBUG("%s: prefetch fired for split %d\n", __func__, next_id);
                            }
                            break;
                        }
                    }
                    if (fired) {
                        fn_event_record(prefetch_event, prefetch_stream);
                        prefetch_pending    = true;
                        prefetch_split_id   = next_id;
                    }
                }
            }
        }
```

- [ ] **Step 4: 全量构建（CUDA + Go），确认编译通过**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\rebuild_windows.ps1
```

Expected: 全量构建成功，`dist\ollama-windows-amd64\` 下有 `ollama.exe` 和 `lib\ollama\cuda_v13\ggml-cuda.dll`。

- [ ] **Step 5: 冒烟测试 — 不开预取，确认 Plan A 行为不变**

```
set OLLAMA_MOE_GPU_LAYERS=-1
set OLLAMA_MOE_PINNED=1
set OLLAMA_DEBUG=1
ollama serve
```

加载 Qwen3-Coder-Next 80B Q4_K_M，发一条短请求，检查日志中：
- `moe pinned: registered CPU-MoE buffer` 仍然出现
- 无 `prefetch enabled` 日志（预取未开）
- 无 crash

- [ ] **Step 6: Commit**

```
git add ml/backend/ggml/ggml/src/ggml-backend.cpp
git commit -m "feat: implement MoE lookahead prefetch in ggml_backend_sched_compute_splits"
```

---

## Task 6: Benchmark — 验证 Plan B 性能

**Files:**
- Test data: `test/moe-split/qwen3-coder-next_moe-split-plan-b.json`（新增）

- [ ] **Step 1: 启动 ollama serve，开启 Plan B**

```
set OLLAMA_MOE_GPU_LAYERS=-1
set OLLAMA_MOE_PINNED=1
set OLLAMA_MOE_PREFETCH=1
set OLLAMA_DEBUG=1
ollama serve
```

确认日志出现：
```
MoE prefetch enabled (stream=0x... event=0x...)
```

如果日志中出现 `prefetch enabled` 但随即有 `prefetch fired` 记录，说明流水线在运行。

- [ ] **Step 2: 在测试机器上手动运行 benchmark**

确认 `ollama serve` 已运行，然后在测试机器上执行：

```
bench-sweep.exe run -model qwen3-coder-next -name moe-split-plan-b -sizes 1024 -batch-size 1024
```

配置（默认值）：warmup=4，epochs=6，batch-size=1024，input tokens=1024，max_tokens=16。

命令完成后将生成的结果 JSON 复制到 `test/moe-split/qwen3-coder-next_moe-split-plan-b.json`。

- [ ] **Step 3: 比对结果**

| 测试 | prefill 均值 | 目标 |
|---|---|---|
| baseline（已有） | 2096.3 ms | — |
| Plan A（已有） | 1391.7 ms | — |
| **Plan B（本次）** | ? | **< 1360 ms** |

成功标准：Plan B prefill 均值 < 1360 ms，CV < 5%，gen_tps 不低于 18.50 t/s。

- [ ] **Step 4: 写实验报告**

创建 `docs/perf/2026-04-16-moe-split-phase2-plan-b-experiment-report.md`，参照 Plan A 报告格式，包含：
- 三组对比表（baseline / Plan A / Plan B）
- 各 epoch 明细
- 分析（是否达到预期，copy/compute 重叠情况）

- [ ] **Step 5: Commit 测试数据和报告**

```
git add test/moe-split/qwen3-coder-next_moe-split-plan-b.json
git add docs/perf/2026-04-16-moe-split-phase2-plan-b-experiment-report.md
git commit -m "docs: add Phase 2 Plan B experiment report"
```

---

## 注意事项

**proc address 查找频率问题**

Task 4 和 Task 5 中，`moe_get_proc` 在循环内被多次调用。每次调用都遍历 backend dev 列表做字符串比较。由于 ggml 没有缓存 proc address 的标准机制，且查找次数固定（每次 compute_splits 调用查找约 10 次，每次查询 `n_backends` 个 dev），对性能影响可忽略不计（ns 级）。若将来需要优化，可在函数开头一次性查出所有函数指针存入局部变量。

**`GGML_LOG_DEBUG` 在 release build 中的开销**

`GGML_LOG_DEBUG` 在 release build 中仍然会执行字符串格式化（除非 ggml 定义了 no-op 宏）。如果日志频繁出现性能问题，可改为 `GGML_DEBUG >= 2` 的条件判断。观察 `ggml-backend.cpp:688` 中 `sched->debug` 字段的用法作为参考。

**全量预取 vs 精确预取的取舍**

本计划采用全量预取（每层传输全部 ~975 MiB expert 权重），实际激活量约为 54%（~525 MiB）。若实验结果显示全量预取无显著收益，可考虑：
1. 等待 compute N 完成后读取 ids_tensor，对 N+1 层做精确预取（比全量慢但传输量小）
2. 放弃 Plan B
