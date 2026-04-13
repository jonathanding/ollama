# Native Backend Per-Node Timing — Design Spec

**Date:** 2026-04-13
**Branch:** `feature/native-perf-tracing`
**Status:** Draft
**Replaces:** Eval callback tracing (2026-03-19 spec)

## Summary

Replace the `ggml_backend_sched_eval_callback` mechanism with backend-native per-node timing. The eval callback forces `ggml_backend_synchronize()` after every node, destroying GPU pipeline overlap and causing 2x+ slowdown. Native timing uses each backend's own instrumentation (Vulkan GPU timestamp queries, CPU `clock_gettime`) with near-zero overhead (<5%).

**Scope:** Vulkan + CPU backends. CUDA not supported in this iteration (timing_ns stays NULL, normal execution).

**Constraint:** JSONL output format unchanged — existing trace-analyzer and web visualizer work with zero modification.

---

## §1 Architecture Overview

### Data Flow

```
+---------------- C layer (inside graph compute) ----------------+
|                                                                 |
|  Vulkan backend                   CPU backend                   |
|  +------------------------+       +------------------------+    |
|  | vkCmdWriteTimestamp     |       | clock_gettime before   |    |
|  | (existing query pool)   |       | ggml_compute_forward   |    |
|  |                        |       | clock_gettime after     |    |
|  | after graph completes:  |       |                        |    |
|  | delta = ts[i] - ts[i-1]|       | delta = after - before  |    |
|  | store in backend buffer |       | store in backend buffer |    |
|  +------------------------+       +------------------------+    |
|            |                               |                    |
|            +---------------+---------------+                    |
|                            v                                    |
|              backend-internal timing arrays                     |
+-----------------------------------------------------------------+
                             |
                             v  CGO read (after Go-side sync)
+---------------- Go layer -------------------------------+
|                                                          |
|  1. Read timing via ggml_backend_sched_get_split_timing  |
|  2. Read node metadata via ggml_node_get_info            |
|  3. Generate OpEvent[] -> append to JSONL buffer         |
|  4. pass_start/pass_end via time.Now() (rides existing   |
|     sync points, zero behavior change)                   |
|  5. Flush -> background goroutine writes file            |
|                                                          |
+----------------------------------------------------------+
                             |
                             v
              trace_<id>_<ts>.jsonl (format unchanged)
                             |
                             v
              trace-analyzer / web visualizer (zero changes)
```

### Enable/Disable Control

- `OLLAMA_TRACE_DIR` unset -> Go side does not call `ggml_backend_sched_set_timing()` -> `timing_enabled` stays false -> C layer skips timing -> **zero overhead**
- `OLLAMA_TRACE_DIR=/path` -> Go side enables timing -> backends record per-node data

### Unsupported Backend Behavior

CUDA or other backends: `get_split_timing()` returns zeros. Go side sees 0 and skips that node — no OpEvent emitted. JSONL contains only Vulkan and CPU data; no incorrect data.

---

## §2 C Layer Changes

### 2.1 ggml_backend_sched Changes

**Remove:**
- `ggml_backend_sched_eval_callback callback_eval`
- `void * callback_eval_user_data`
- The entire callback branch in `compute_splits()` (lines 1621-1653)
- `ggml_backend_sched_set_eval_callback()` function and type declaration

**Add:**

```c
// ggml_backend_sched struct — single new field
bool timing_enabled;   // set by Go side
```

**Modify `compute_splits()`:** Delete the `if (sched->callback_eval)` branch. The remaining path is unchanged — one `ggml_backend_graph_compute_async()` call per split. Timing data collection happens inside each backend's `graph_compute` implementation, gated on each backend's own `timing_enabled` flag.

#### timing_enabled 传播机制

backends 没有 sched 的反向指针（`ggml_backend` struct 只有 `void* context`，无 sched 字段），因此 `sched->timing_enabled` 无法直接被 backend 读取。解决方案：**每个 backend 维护独立的 `timing_enabled` 标志**，由 sched 统一传播：

```c
// ggml-backend-impl.h — ggml_backend struct 新增字段
bool timing_enabled;

// ggml-backend.cpp — 新增函数
void ggml_backend_set_timing(ggml_backend_t backend, bool enabled) {
    backend->timing_enabled = enabled;
}

// ggml_backend_sched_set_timing 传播到所有 backends
void ggml_backend_sched_set_timing(ggml_backend_sched_t sched, bool enabled) {
    sched->timing_enabled = enabled;
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_set_timing(sched->backends[i], enabled);
    }
}
```

各 backend 的 `graph_compute` 实现通过 `backend->timing_enabled` 判断是否收集 timing 数据。这避免了修改 `graph_compute_async` 的函数签名，也避免了给 backend struct 添加 sched 反向指针。

**注意：** `backend->timing_enabled` 是 per-physical-device 全局的。Go 层的 `backends` map 是 `sync.OnceFunc` 初始化的单例（ggml.go:48-70），所有使用同一物理设备的 `Backend` 实例共享同一 `ggml_backend_t`。在 Ollama 的架构中，同一时刻通常只有一个模型在一个设备上运行，因此不存在冲突。如果未来支持多模型并行推理且需要独立的 timing 开关，需改为 per-sched 标志。

### 2.2 New Public API (ggml-backend.h)

```c
// Enable/disable timing collection (propagates to all backends)
void ggml_backend_sched_set_timing(ggml_backend_sched_t sched, bool enabled);

// Per-backend timing flag (called by sched_set_timing, or directly for testing)
void ggml_backend_set_timing(ggml_backend_t backend, bool enabled);

// Query split structure (read-only, valid after graph_compute)
int  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
int  ggml_backend_sched_get_split_start(ggml_backend_sched_t sched, int split_id);
int  ggml_backend_sched_get_split_n_nodes(ggml_backend_sched_t sched, int split_id);
int  ggml_backend_sched_get_split_backend_id(ggml_backend_sched_t sched, int split_id);

// Read timing data for a split (call after synchronize)
// Returns number of nodes written. timing_out allocated by caller.
int  ggml_backend_sched_get_split_timing(
         ggml_backend_sched_t sched, int split_id,
         uint64_t * timing_out, int capacity);

// Extract node metadata (avoids duplicating extraction logic in Go)
struct ggml_node_info {
    const char * op_name;
    const char * tensor_name;
    int64_t      shape[4];
    const char * dtype_name;
    const char * backend_name;
    const char * src_names[GGML_MAX_SRC];
    int          n_srcs;
};

void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out);
```

### 2.3 Vulkan Backend Changes

Existing infrastructure: query pool, `vkCmdWriteTimestamp`, timestamp delta calculation in the perf logger code path.

**关键约束：** 现有的 `GGML_VK_PERF_LOGGER` 实现在 `graph_compute` **内部** 调用 `waitForFences` + `getQueryPoolResults`（ggml-vulkan.cpp:13333-13353），使整个 compute 变成同步的。新的 timing 机制**绝不能**这样做——必须拆分为两步：

#### 第一步：异步 timestamp 插入（graph_compute 内部）

- 当 `backend->timing_enabled` 为 true 时，在每个 node 的 dispatch 后插入 `vkCmdWriteTimestamp`
- 复用现有的 query pool 和 `query_nodes[]` 数组
- **不调用 `waitForFences`**，command buffer 正常提交，GPU 异步执行
- 这一步在 `ggml_backend_graph_compute_async()` 内完成

#### 第二步：延迟读取（Go-side sync 之后）

- 新增函数 `ggml_vk_collect_timing(backend, uint64_t* timing_out, int capacity)`
- 由 `ggml_backend_sched_get_split_timing()` 在内部调用（对该 split 的 backend dispatch）
- 此时 Go 层已经调用了 `ggml_backend_sched_synchronize()`，所有 fences 已 signaled
- 调用 `getQueryPoolResults` 读回 timestamp 值
- 计算 delta：`timing_out[i] = (timestamps[i+1] - timestamps[i]) * timestampPeriod`
- 结果存入调用者分配的 `timing_out` 数组

#### 存储和生命周期

- 在 `ggml_backend_vk_context` struct（ggml-vulkan.cpp:1678）中新增 `std::vector<uint64_t> node_timing_ns`，与现有 `query_pool`、`query_nodes`（std::vector）同级
- 在第一步 timestamp 插入时，`node_timing_ns.resize(num_queries)` 与 `query_nodes` 同步 resize
- 在第二步 `ggml_vk_collect_timing()` 中填充 delta 值
- context 销毁时 `std::vector` 自动释放，无需手动管理
- timestamp 数据在 `ggml_backend_sched_synchronize()` 后可读，在下一次 `graph_compute` 时被覆写

**Changes (~100 lines in ggml-vulkan.cpp):**

- Gate timestamp query insertion on `backend->timing_enabled`（与现有 `vk_perf_logger_enabled` 检查并行）
- 新增 `ggml_vk_collect_timing()` 函数：读 query pool + 计算 delta
- `GGML_VK_PERF_LOGGER` stderr output 保持独立——两者可以共存

### 2.4 CPU Backend Changes

**Changes (~30 lines in ggml-cpu.c):**

CPU compute 使用线程池实现数据并行：每个 node 由多个线程并行处理同一个 node 的不同数据分片，线程之间通过 `ggml_barrier()` 同步（ggml-cpu.c:2961）。单个线程的 `ggml_compute_forward()` 返回只表示该线程完成了自己的分片，而非整个 node 完成。

因此 timing 必须在 **thread 0 上**、**barrier-to-barrier** 测量，才能捕获整个 node 的实际 wall time：

```c
// Inside ggml_graph_compute_thread() — thread 0 only
// ggml-cpu.c per-node loop (lines 2940-2963)
for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
    struct ggml_tensor * node = cgraph->nodes[node_n];

    uint64_t t0 = 0;
    if (ith == 0 && backend->timing_enabled) {
        t0 = ggml_time_us();
    }

    ggml_compute_forward(&params, node);

    ggml_barrier(state->threadpool);  // 所有线程完成此 node

    if (ith == 0 && backend->timing_enabled) {
        cpu_ctx->node_timing_ns[node_n] = (ggml_time_us() - t0) * 1000;
    }
}
```

**关键点：**
- `ggml_barrier()` 已存在于每个 node 之间（line 2961），不是新增的同步点
- 只有 thread 0 执行 timing 代码，其他线程零额外开销
- `t0` 在 barrier 之前采样（包含 thread 0 自身的 compute 时间），`t_end` 在 barrier 之后采样（所有线程都完成）
- 测量的是 node 级 wall time（barrier-to-barrier），精确反映该 node 占用 CPU 线程池的实际时间

#### 存储

在 `ggml_backend_cpu_context` struct（ggml-cpu.cpp:99-108）中新增两个字段：

```c
struct ggml_backend_cpu_context {
    // ... existing fields ...
    uint64_t * node_timing_ns;   // per-node timing array, NULL when timing disabled
    int        timing_capacity;  // allocated capacity (number of nodes)
};
```

- 在 `graph_compute` 开头，若 `backend->timing_enabled`，检查 capacity 并按需 realloc：`if (cgraph->n_nodes > cpu_ctx->timing_capacity) { ... }`
- `ggml_backend_cpu_free()`（line 116）中释放：`delete[] cpu_ctx->node_timing_ns;`
- timing 未启用时两个字段保持为 NULL/0，零开销

Overhead: ~20ns per `ggml_time_us()` call (thread 0 only). 不添加任何新的同步点，不改变线程池行为。

### 2.5 llama.cpp API Wrappers

Thin wrappers in `llama.h` / `llama-context.cpp` for LlamaRunner access:

```c
void llama_context_enable_timing(struct llama_context * ctx, bool enabled);
int  llama_context_get_n_splits(struct llama_context * ctx);
int  llama_context_get_split_start(struct llama_context * ctx, int split_id);
int  llama_context_get_split_n_nodes(struct llama_context * ctx, int split_id);
int  llama_context_get_split_backend_id(struct llama_context * ctx, int split_id);
int  llama_context_get_split_timing(struct llama_context * ctx, int split_id,
                                     uint64_t * out, int capacity);
struct ggml_cgraph * llama_context_get_graph(struct llama_context * ctx);
```

Each is a one-line delegation to the corresponding `ggml_backend_sched_*` function via `ctx->sched`.

### 2.6 Split Graph Structure (Reference)

Splits are contiguous, non-overlapping ranges in the full graph:

```
split[k].i_end == split[k+1].i_start

Example (Vulkan + CPU offload):
  Split 0: backend=CPU,    i_start=0,    i_end=100
  Split 1: backend=Vulkan, i_start=100,  i_end=9900
  Split 2: backend=CPU,    i_start=9900, i_end=10000
```

Each split's sub-graph is a view: `split->graph.nodes[j]` points to `full_graph->nodes[split->i_start + j]`. Timing data is collected per-split, then mapped to global node indices via `i_start` offset.

Timing data retrieval happens after Go-side `ggml_backend_sched_synchronize()` — at this point all backends have completed and all timing data is readable. No additional sync points are introduced in `compute_splits`.

---

## §3 Go Layer Changes

### 3.1 New Profiler Package (`llm/profiler/`)

Delete all existing files and rewrite. The core change: **push model** (C callback pushes events to Go) becomes **pull model** (Go pulls timing data after sync).

**`profiler.go` — Interface and types:**

```go
// TraceWriter replaces TraceCollector.
// Push model (old): C callback -> RecordTensorStart/End
// Pull model (new): Go calls WriteOps after sync
type TraceWriter interface {
    WriteOps(ops []OpEvent)
    WritePassStart(passID int, nTokens int)
    WritePassEnd(passID int)
    Flush(requestID string, model string) error
    Close() error
}

// OpEvent — unchanged from old spec, JSONL schema compatible
type OpEvent struct {
    Type     string   `json:"type"`
    PassID   int      `json:"pass"`
    SeqID    int      `json:"seq"`
    Op       string   `json:"op"`
    Name     string   `json:"name"`
    SrcNames []string `json:"srcs"`
    OutShape []int64  `json:"shape"`
    DType    string   `json:"dtype"`
    Backend  string   `json:"backend"`
    TStart   int64    `json:"t_start"`
    TEnd     int64    `json:"t_end"`
}

// TensorInfo — unchanged
type TensorInfo struct {
    Op       string
    Name     string
    SrcNames []string
    OutShape []int64
    DType    string
    Backend  string
}
```

**`jsonl.go` — JSONLWriter:**

- `WriteOps(ops []OpEvent)`: assigns sequential SeqID, serializes each op to JSON, appends to `[][]byte` buffer
- `WritePassStart/End`: same as old implementation (JSON serialize + append)
- `Flush`: same as old (background goroutine writes file)
- Thread-safe via `sync.Mutex`

**`noop.go` — NoopWriter:**

All methods are empty. Used when `OLLAMA_TRACE_DIR` is unset. Zero overhead.

**Factory:**

```go
func NewWriter(outDir string) TraceWriter {
    if outDir == "" {
        return &NoopWriter{}
    }
    return newJSONLWriter(outDir)
}
```

### 3.2 Timestamp Reconstruction

Backend-native timing provides per-node **elapsed duration** (delta), not absolute timestamps. The JSONL format requires `t_start` / `t_end`. Reconstruction in Go using cumulative offset:

```go
// Inside CollectTiming(), after reading all splits
baseTime := passStartTime.UnixNano()
cursor := baseTime
for i := range events {
    events[i].TStart = cursor
    events[i].TEnd = cursor + int64(elapsedNs[i])
    cursor = events[i].TEnd
}
```

对于 CPU backend，节点严格顺序执行（barrier 保证），linearization 完全准确。

对于 Vulkan backend，GPU 存在 pipeline overlap（节点 A 和 B 可能部分并行执行）。`vkCmdWriteTimestamp`（`VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT`）记录的是序列化的完成时间点，delta[i] = timestamp[i+1] - timestamp[i] 反映的是两个完成点之间的 wall time 间隔，而非单个 kernel 的独立执行时间。因此：

- `sum(elapsed)` = 整个序列的 GPU wall time（首节点开始到末节点完成），**不是** kernel 独立执行时间之和
- 个别节点的 delta 是近似值：pipeline overlap 下，后续节点的 delta 可能小于其实际执行时间（因为部分执行与前一节点重叠）
- **相对比例仍然准确**：delta 较大的节点确实是瓶颈，timeline 可视化正确反映热点分布
- 这与旧的 eval callback 机制行为一致（旧机制强制同步，消除了 overlap，反而使 delta 失真为非 pipeline 状态下的时间）

### 3.3 CGO Bridge Files

**`ml/backend/ggml/timing_bridge.go`** — OllamaRunner path:

```go
func (b *Backend) EnableTiming(enabled bool) {
    C.ggml_backend_sched_set_timing(b.sched, C.bool(enabled))
}

func (b *Backend) CollectTiming(graph *C.ggml_cgraph, passStartTime time.Time) []profiler.OpEvent {
    nSplits := int(C.ggml_backend_sched_get_n_splits(b.sched))
    var events []profiler.OpEvent

    for s := 0; s < nSplits; s++ {
        nNodes := int(C.ggml_backend_sched_get_split_n_nodes(b.sched, C.int(s)))
        iStart := int(C.ggml_backend_sched_get_split_start(b.sched, C.int(s)))

        buf := make([]C.uint64_t, nNodes)
        C.ggml_backend_sched_get_split_timing(b.sched, C.int(s), &buf[0], C.int(nNodes))

        for j := 0; j < nNodes; j++ {
            elapsed := uint64(buf[j])
            if elapsed == 0 { continue }

            node := graphNodeAt(graph, iStart+j)  // C pointer arithmetic helper
            var info C.struct_ggml_node_info
            C.ggml_node_get_info(node, &info)

            events = append(events, profiler.OpEvent{
                Type:    "op",
                Op:      C.GoString(info.op_name),
                Name:    C.GoString(info.tensor_name),
                // ... shape, dtype, backend, srcs extraction ...
                TEnd:    int64(elapsed),  // absolute timestamps assigned later
            })
        }
    }

    // Reconstruct absolute timestamps
    cursor := passStartTime.UnixNano()
    for i := range events {
        events[i].TStart = cursor
        events[i].TEnd = cursor + events[i].TEnd
        cursor = events[i].TEnd
    }

    return events
}
```

**`llama/timing_bridge.go`** — LlamaRunner path:

Same pattern, but calls `llama_context_*` wrappers instead of `ggml_backend_sched_*` directly. Metadata extraction uses the same `ggml_node_get_info()` via the graph pointer from `llama_context_get_graph()`.

---

## §4 Runner Integration

### 4.1 OllamaRunner (`runner/ollamarunner/runner.go`)

**Remove:**
- `prof profiler.TraceCollector` field
- `import "github.com/ollama/ollama/llm/profiler"` (re-add with new package)
- All `RecordPassStart/End`, `SetEvalCallback`, `Flush`, `Close` calls
- `evalCallbackSetter` interface

**Add:**

```go
// Field
traceWriter profiler.TraceWriter

// Initialization (in loadModel or equivalent)
s.traceWriter = profiler.NewWriter(envconfig.TraceDir())
if _, ok := s.traceWriter.(*profiler.JSONLWriter); ok {
    if tb, ok := s.model.Backend().(interface{ EnableTiming(bool) }); ok {
        tb.EnableTiming(true)
    }
}

// In computeBatch:
passStart := time.Now()
s.traceWriter.WritePassStart(activeBatch.id, len(batchInputs))
activeBatch.ctx.ComputeWithNotify(cb, activeBatch.modelOutput)
outputs := activeBatch.modelOutput.Floats()   // sync happens here
s.traceWriter.WritePassEnd(activeBatch.id)

// Pull timing after sync
if ct, ok := s.model.Backend().(interface{
    CollectTiming(*C.ggml_cgraph, time.Time) []profiler.OpEvent
}); ok {
    if events := ct.CollectTiming(graph, passStart); len(events) > 0 {
        s.traceWriter.WriteOps(events)
    }
}

// On request completion:
s.traceWriter.Flush(requestID, s.modelPath)
```

### 4.2 LlamaRunner (`runner/llamarunner/runner.go`)

**Remove:** Same cleanup as OllamaRunner (prof field, import, RecordPass*, SetEvalCallback, Flush).

**Add:**

```go
// Field
traceWriter profiler.TraceWriter

// Initialization
s.traceWriter = profiler.NewWriter(envconfig.TraceDir())
if _, ok := s.traceWriter.(*profiler.JSONLWriter); ok {
    s.lc.EnableTiming(true)
}

// In processBatch:
passStart := time.Now()
s.traceWriter.WritePassStart(batchID, batch.NumTokens())
s.lc.Decode(batch)
s.lc.Synchronize()   // timing data readable after this
s.traceWriter.WritePassEnd(batchID)

// Pull timing
if events := s.lc.CollectTiming(passStart); len(events) > 0 {
    s.traceWriter.WriteOps(events)
}

// On request completion:
s.traceWriter.Flush(requestID, s.modelPath)
```

### 4.3 Sync Point Analysis (No Behavior Change)

**LlamaRunner:** `Synchronize()` is already called after every `Decode()` when `numOutputs > 0`. Timing read happens after this existing sync. No new sync added.

**OllamaRunner:** `Floats()` triggers deferred `ggml_backend_sched_synchronize()`. Timing read happens after `Floats()`. No new sync added. `WritePassEnd` is moved after `Floats()` (from before) to capture accurate wall time — this only moves a `time.Now()` call, no behavioral change.

---

## §5 Cleanup, Testing, and Documentation

### 5.1 Files to Delete (6 files, ~501 lines)

| File | Lines | Content |
|------|-------|---------|
| `llm/profiler/profiler.go` | 62 | Old TraceCollector interface |
| `llm/profiler/jsonl.go` | 120 | Old JSONLTraceBuffer (callback push model) |
| `llm/profiler/noop.go` | 12 | Old NoopCollector |
| `llm/profiler/profiler_test.go` | 97 | Old tests |
| `llama/profiler_bridge.go` | 109 | Old LlamaRunner CGO eval callback bridge |
| `ml/backend/ggml/profiler_bridge.go` | 101 | Old OllamaRunner CGO eval callback bridge |

### 5.2 Lines to Remove from Existing Files

| File | Remove |
|------|--------|
| `runner/llamarunner/runner.go` | `prof` field, `batchID` field, profiler import, RecordPassStart/End, SetEvalCallback, Flush calls |
| `runner/ollamarunner/runner.go` | `prof` field, profiler import, RecordPassStart/End, evalCallbackSetter interface + dynamic check, Flush, Close calls |
| `ml/backend/ggml/ggml.go` | `profilerHandle cgo.Handle` field + cleanup in Close() |
| `llama/llama.go` | `profilerHandle cgo.Handle` field |
| `ggml-backend.cpp` | `callback_eval` / `callback_eval_user_data` fields, callback branch in compute_splits (lines 1621-1653), `set_eval_callback` function |
| `ggml-backend.h` | `ggml_backend_sched_eval_callback` type, `ggml_backend_sched_set_eval_callback` declaration |

### 5.3 New Files

| File | Content |
|------|---------|
| `llm/profiler/profiler.go` | TraceWriter interface + OpEvent/TensorInfo (JSONL schema unchanged) |
| `llm/profiler/jsonl.go` | JSONLWriter: WriteOps batch input + async Flush |
| `llm/profiler/noop.go` | NoopWriter |
| `llm/profiler/profiler_test.go` | Tests for new TraceWriter |
| `ml/backend/ggml/timing_bridge.go` | OllamaRunner CGO: EnableTiming + CollectTiming |
| `llama/timing_bridge.go` | LlamaRunner CGO: EnableTiming + CollectTiming via llama_context wrappers |

### 5.4 C Layer File Changes

| File | Change |
|------|--------|
| `ggml-backend.h` | Add timing API declarations, `ggml_node_info` struct, `ggml_node_get_info` |
| `ggml-backend.cpp` | Add `timing_enabled` to sched; delete callback branch; implement timing API |
| `ggml-vulkan.cpp` | Refactor perf logger: store per-node delta in array (~80 lines) |
| `ggml-cpu.c` | Add per-node clock_gettime wrapper (~30 lines) |
| `llama.h` | Add `llama_context_*_timing` wrapper declarations |
| `llama-context.cpp` | Implement wrappers (~7 one-line delegations) |
| `llama/patches/0018-ggml-Add-batch-size-hint.patch` | 重新生成 — 当前 line 103 引用 `sched->callback_eval`，删除 callback 后 patch 会失败 |

### 5.5 Testing Strategy

**Unit tests:**

| Test | Validates |
|------|-----------|
| `llm/profiler/profiler_test.go` | WriteOps -> Flush -> JSONL format correct, field completeness, pass events |

**Integration tests (manual):**

| Scenario | Expected |
|----------|----------|
| No `OLLAMA_TRACE_DIR`, run inference | Zero overhead, no output, no crash |
| `OLLAMA_TRACE_DIR` + CPU-only model | JSONL produced, per-node times reasonable |
| `OLLAMA_TRACE_DIR` + Vulkan model | JSONL produced, GPU timestamps accurate |
| Feed JSONL to trace-analyzer | DAG, Timeline, Heatmap, Replay all render correctly |
| Compare old vs new JSONL | Format compatible, all fields present |

**Performance verification:**

| Test | Target |
|------|--------|
| `ollama-bench` without TRACE_DIR | Baseline token/s (should be identical) |
| `ollama-bench` with TRACE_DIR | token/s degradation < 5% (old approach: > 50%) |

### 5.6 Documentation Updates

| Document | Change |
|----------|--------|
| `README.md` | Update tracing architecture description |
| `docs/debugging-and-profiling.md` | Update eval callback sections, note replacement |
| `docs/internals/01-eval-callback-tracing.md` | Mark as historical or rewrite for native timing |
| `envconfig/config.go` comment | Update `OLLAMA_TRACE_DIR` description |

**No changes needed:**
- `tools/trace-analyzer/README.md` — user-facing usage unchanged
- `tools/trace-analyzer/RELEASE-README.md` — same
- `docs/superpowers/specs/2026-03-19-*` — kept as historical record

---

## §6 Risk and Open Questions

### Risks

| Risk | Mitigation |
|------|------------|
| Upstream ggml merge conflicts (sched struct change) | Changes are minimal (1 bool field + delete 2 fields). Split API functions are additive. |
| Vulkan timestamp accuracy on some GPUs | `timestampPeriod` is spec-mandated. Fall back to wall-clock if `timestampValidBits == 0`. |
| Graph node pointers invalid after next compute | CollectTiming is called after sync but before the next `graph_compute` (which calls `split_graph` → overwrites splits). `sched_reset()` 本身不清除 splits（已验证：只重置 hash_set 和 flags），数据安全。 |
| CGO 调用开销（~150ns/call × 500 nodes）| 当前设计每个 node 一次 `ggml_node_get_info` CGO 调用。若成为瓶颈，可改为 batch API：`ggml_backend_sched_get_split_timing_batch()` 一次返回整个 split 的 timing + metadata，将 500 次 CGO 调用降为 ~10 次（每 split 一次）。Phase 1 先用简单的 per-node 调用，性能不达标再优化。 |

### Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Per-pass timing changes behavior? | No. Both runners already sync after each batch. time.Now() rides existing sync points. |
| OllamaRunner WritePassEnd placement? | Move after Floats() — only moves a time.Now() call, no behavioral change. |
| LlamaRunner support? | Yes, via llama_context wrapper functions (same pattern as old set_eval_callback). |
| JSONL format compatibility? | Fully compatible. t_start/t_end reconstructed from cumulative elapsed delta. |
| Missing time (kernel-only vs wall)? | By design. Per-pass wall time captures total; per-node captures kernel. Gap = non-compute overhead. |
