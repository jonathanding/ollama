# Ollama Execution Profiling & Tracing System

**Date**: 2026-03-19
**Status**: Draft

---

## Background & Motivation

Ollama wraps llama.cpp (and a newer native Go inference engine) to serve LLM inference. When debugging performance, understanding bottlenecks, or analyzing data flow, the internal execution is a black box: a request comes in and a response comes out, but the details are invisible — which operators ran, how long each took, whether data was on CPU or GPU, where data copies occurred, or how operators depend on each other.

This project adds a **non-intrusive, opt-in profiling/tracing system** that instruments the execution path and captures a detailed trace per inference request. Trace data is accumulated in memory during inference to minimize I/O overhead, then flushed asynchronously to disk after each request completes.

### Goals

- Capture the **computation graph**: which operators execute, in what order, and with what data dependencies
- Measure **per-operator performance**: approximate execution time, CPU vs. GPU assignment
- Track **data copies**: copies between backends (CPU↔GPU), their sizes and timing (via GGML copy op nodes)
- Record **tensor metadata**: shapes, data types, names (e.g., `blk.3.attn_q`)
- Produce output suitable for **DAG visualization**: nodes = operators, edges = data flow with cost annotations
- **Zero overhead when disabled** — null callback registered in GGML scheduler, which skips the callback entirely

### Non-Goals

- Exact GPU execution time per operator (accurate GPU timing requires CUDA events or Vulkan timestamp queries; CPU-dispatch timing is sufficient for identifying hotspots in Phase 1)
- Full memory profiling (peak VRAM, per-layer allocation) — not in scope for Phase 1
- Profiling OllamaRunner or MLX — Phase 2

---

## Future Visualization (Phase 2 Feature)

A separate web application will consume the trace files and render an interactive computation graph:

- **Graph view**: DAG where nodes are operators and edges are data tensors; node color encodes backend (CPU/GPU); node size encodes execution time; edge width encodes tensor size
- **Timeline view**: per-layer timing breakdown across a full request
- **Copy heatmap**: highlights significant CPU↔GPU transfers
- **Multi-run comparison**: overlay traces from multiple requests to spot variance

---

## Architecture Overview

Two key insights about the codebase shape this design:

1. **LlamaRunner** (the current default path) calls `lc.Decode()` — a single CGO call that executes the entire GGML compute graph inside C++. Go cannot observe individual operators without a C-side hook.

2. **GGML already has** `ggml_backend_sched_eval_callback`, a per-node callback that fires for every compute node during graph execution, including `GGML_OP_CPY`/`GGML_OP_DUP` copy nodes. It works for all backends (CPU, CUDA, Vulkan). Ollama simply does not expose it yet.

```
┌──────────────────────────────────────────────────────────────┐
│                   LlamaRunner (Go)                           │
│  processBatch()                                              │
│    ├─ profiler.RecordPassStart(passID, nTokens)              │
│    ├─ lc.Decode(batch)          ← single CGO call            │
│    │    └─ per GGML node: C eval callback fires              │
│    │         └─ callback reads user_data → collector         │
│    │              └─ collector.RecordOp() / RecordCopy()     │
│    ├─ lc.Synchronize()                                       │
│    └─ profiler.RecordPassEnd(passID)                         │
│                                                              │
│  On request completion:                                      │
│    collector.Flush(requestID)   ← async goroutine            │
└──────────────────────────────────────────────────────────────┘
           │ CGO
┌──────────────────────────────────────────────────────────────┐
│              llama/llama.go  (CGO wrapper)                   │
│  Context.SetEvalCallback(collector *profiler.Collector)      │
│    └─ stores Go pointer in C.ggml_profiler_ctx               │
│    └─ C.llama_context_set_eval_callback(c.c, bridge, ctx)    │
└──────────────────────────────────────────────────────────────┘
           │ new C API (~10 lines)
┌──────────────────────────────────────────────────────────────┐
│         vendored llama.cpp                                   │
│  llama_context_set_eval_callback(ctx, cb, user_data)         │
│    └─ ggml_backend_sched_set_eval_callback(ctx->sched, ...)  │
│         fires: cb(tensor, ask=true)   before op dispatch     │
│                cb(tensor, ask=false)  after op dispatch      │
└──────────────────────────────────────────────────────────────┘
```

---

## Scope

### Phase 1 (this document)

Instrument **LlamaRunner** — the default path used when running `ollama serve`. Targets CUDA and Vulkan GPU backends.

### Phase 2 (future)

Instrument **OllamaRunner**. The `TraceCollector` Go interface accepts events from both paths; Phase 2 needs only a new event source. No format or visualization changes required.

---

## Component Design

### 1. C++ Patch (vendored llama.cpp)

**Files**: `llama/llama.h`, `llama/llama.cpp`
**Total change**: ~10 lines

```c
// llama.h — new declaration
void llama_context_set_eval_callback(
    struct llama_context * ctx,
    ggml_backend_sched_eval_callback cb,
    void * user_data
);
```

```cpp
// llama.cpp — new implementation
// ctx->sched is a smart pointer in llama.cpp; use .get() to obtain the raw pointer.
// Verify the exact member access pattern against the vendored llama-context.h before
// final implementation.
void llama_context_set_eval_callback(
    struct llama_context * ctx,
    ggml_backend_sched_eval_callback cb,
    void * user_data
) {
    ggml_backend_sched_set_eval_callback(ctx->sched.get(), cb, user_data);
}
```

**Disabling**: passing `cb = NULL` causes GGML to skip the callback entirely (`ggml_backend_sched_graph_compute_async` checks `sched->callback_eval != NULL` before invoking). This ensures zero overhead when profiling is disabled.

### 2. CGO Wrapper (`llama/llama.go`)

Go cannot pass function pointers directly to C. The pattern: a static C function receives `user_data` (which carries a cgo.Handle to the Go collector), then calls back into Go.

```c
/* In the CGO preamble of llama.go */
#include "runtime/cgo.h"

static _Bool profilerEvalCallbackBridge(
    struct ggml_tensor* t, _Bool ask, void* user_data
) {
    /* user_data is a uintptr_t holding a cgo.Handle value */
    GoProfilerCallback((GoUintptr)(uintptr_t)user_data, (void*)t, ask);
    return ask; /* true = also receive the post-dispatch call */
}
```

```go
//export GoProfilerCallback
func GoProfilerCallback(handle C.GoUintptr, tensorPtr unsafe.Pointer, ask C.bool) {
    h := cgo.Handle(handle)
    collector := h.Value().(*profiler.JSONLTraceBuffer)
    collector.HandleTensor(tensorPtr, bool(ask))
}

// SetEvalCallback registers per-node profiling on this context.
// Pass nil collector to disable (zero overhead).
func (c *Context) SetEvalCallback(collector *profiler.JSONLTraceBuffer) {
    if collector != nil {
        h := cgo.NewHandle(collector)
        c.profilerHandle = h  // stored on Context to prevent GC and for later deletion
        C.llama_context_set_eval_callback(c.c, C.profilerEvalCallbackBridge,
            C.GoUintptr(uintptr(h)))
    } else {
        if c.profilerHandle != 0 {
            c.profilerHandle.Delete()
            c.profilerHandle = 0
        }
        C.llama_context_set_eval_callback(c.c, nil, nil)
    }
}
```

**cgo.Handle** is the standard Go mechanism for passing Go values through C without violating the CGO pointer rules (no Go pointers may be stored in C memory). The handle is an integer that `cgo.Handle.Value()` maps back to the Go object.

**Tensor data extracted** (via safe CGO field access):

| Data | C expression | Example |
|---|---|---|
| Operation | `ggml_op_name(t->op)` | `"MUL_MAT"` |
| Tensor name | `t->name` (char[64]) | `"blk.3.attn_q"` |
| Output shape | `t->ne[0..3]` | `[4096, 512, 1, 1]` |
| Data type | `ggml_type_name(t->type)` | `"f16"` |
| Input names | `t->src[i]->name` for `i=0..GGML_MAX_SRC-1`, stop when `t->src[i]==NULL` | `["blk.3.attn_norm", "attn_q.weight"]` |
| Backend | `ggml_backend_buffer_name(t->buffer)` | `"CUDA0"` |

`GGML_MAX_SRC` (defined as 10 in `ggml.h`) is the safe upper bound for the `src[]` array. Each entry must be checked for `NULL` before dereferencing.

### 3. Go Profiler Package (`llm/profiler/`)

New package with three files.

#### `profiler.go` — public interface and event types

```go
// TraceCollector is the unified interface for all instrumentation paths.
// Phase 1: LlamaRunner feeds events via CGO callback.
// Phase 2: OllamaRunner feeds events from Go-layer operator calls.
type TraceCollector interface {
    HandleTensor(tensorPtr unsafe.Pointer, ask bool) // called from CGO bridge
    RecordPassStart(passID int, nTokens int)
    RecordPassEnd(passID int, nNodes int)
    Flush(requestID string, model string) error
    Close() error
}

// OpEvent represents one operator node execution.
type OpEvent struct {
    PassID   int
    SeqID    int      // sequential index within the pass (order of dispatch)
    Op       string   // "MUL_MAT", "FLASH_ATTN_EXT", "CPY", "RMS_NORM", etc.
    Name     string   // tensor name, e.g. "blk.3.attn_q"
    SrcNames []string // input tensor names → become DAG edges
    OutShape []int64
    DType    string   // "f32", "f16", "q4_0", etc.
    Backend  string   // "CPU", "CUDA0", "Vulkan0", etc.
    TStart   int64    // Unix nanoseconds, recorded on ask=true call
    TEnd     int64    // Unix nanoseconds, recorded on ask=false call
}
```

**PassID semantics**: PassID is the batch decode call counter. Each call to `lc.Decode()` is one pass. When LlamaRunner batches multiple sequences together, they share the same PassID. SeqID is a monotonically increasing counter within a single pass (reflects the order GGML dispatches nodes). These two fields together uniquely identify a node event within a request.

**Copy events**: GGML inserts explicit `GGML_OP_CPY` and `GGML_OP_DUP` nodes into the compute graph when data must be copied between backends. These nodes appear in the callback like any other op, with `op="CPY"` or `op="DUP"`. The source and destination backends can be inferred by comparing `t->buffer->buft` against `t->src[0]->buffer->buft`. No separate copy-tracking mechanism is needed; they are captured as regular `OpEvent` records with `op="CPY"`.

#### `jsonl.go` — buffered JSONL writer

`JSONLTraceBuffer` is NOT a global singleton. Each `llama.Context` gets its own collector instance. This avoids concurrency issues — each `Decode()` call writes to its own buffer, and Go's `processBatch()` logic serializes Decode calls with a mutex already.

```go
type JSONLTraceBuffer struct {
    mu      sync.Mutex
    lines   [][]byte  // one serialized JSON line per event
    outDir  string
    passID  int
    seqID   int       // reset each pass
}

// HandleTensor is called from the CGO bridge on the C thread.
// ask=true: record t_start; ask=false: finalize and append the event line.
func (b *JSONLTraceBuffer) HandleTensor(tensorPtr unsafe.Pointer, ask bool) {
    t := (*C.struct_ggml_tensor)(tensorPtr)
    if ask {
        b.mu.Lock()
        b.pending[t] = time.Now().UnixNano()
        b.mu.Unlock()
    } else {
        b.mu.Lock()
        tStart := b.pending[t]  // keyed by C pointer (stable during one Decode call)
        delete(b.pending, t)
        seqID := b.seqID
        b.seqID++
        b.mu.Unlock()

        ev := buildOpEvent(t, b.passID, seqID, tStart, time.Now().UnixNano())
        line, _ := json.Marshal(ev)
        b.mu.Lock()
        b.lines = append(b.lines, line)
        b.mu.Unlock()
    }
}

// Flush hands lines to a background goroutine and clears the buffer.
// Always returns nil; file write errors are logged (not returned) since
// the caller cannot act on them — the inference is already complete.
func (b *JSONLTraceBuffer) Flush(requestID, model string) error {
    b.mu.Lock()
    lines := b.lines
    b.lines = nil
    b.mu.Unlock()

    go func() {
        ts := time.Now().UnixMilli()
        fname := filepath.Join(b.outDir,
            fmt.Sprintf("trace_%s_%d.jsonl", sanitize(requestID), ts))
        f, err := os.Create(fname)
        if err != nil {
            slog.Warn("profiler: failed to create trace file", "path", fname, "err", err)
            return
        }
        defer f.Close()
        for _, line := range lines {
            f.Write(line)
            f.Write([]byte{'\n'})
        }
    }()
    return nil
}
```

**Pending map key**: The C pointer `*C.struct_ggml_tensor` is used as the map key. Tensor pointers are stable for the duration of a single `Decode()` call because GGML allocates tensors from a context arena that is not freed until after compute completes.

#### `noop.go` — zero-overhead default

```go
type NoopCollector struct{}
func (n *NoopCollector) HandleTensor(unsafe.Pointer, bool) {}
func (n *NoopCollector) RecordPassStart(int, int)           {}
func (n *NoopCollector) RecordPassEnd(int, int)             {}
func (n *NoopCollector) Flush(string, string) error         { return nil }
func (n *NoopCollector) Close() error                       { return nil }
```

With `NoopCollector`, `SetEvalCallback(nil)` registers a NULL callback in GGML — the eval path is never invoked.

### 4. LlamaRunner Integration (`runner/llamarunner/runner.go`)

- **`Server` struct**: add `profiler profiler.TraceCollector` field
- **`loadModel()`**: after `llama.NewContextWithModel()`, if `OLLAMA_TRACE_DIR` is set, create `JSONLTraceBuffer` and call `lc.SetEvalCallback(collector)`
- **`processBatch()`**:
  ```go
  s.profiler.RecordPassStart(batchID, batchSize)
  err := s.lc.Decode(batch)
  s.lc.Synchronize()
  s.profiler.RecordPassEnd(batchID, /* node count from graph */)
  ```
- **Completion handler** (per-sequence, in `generateSequenceResponse`):
  - Emit `request_start` event at the start of a completion request
  - Emit `request_end` event + call `s.profiler.Flush(req.ID, s.modelPath)` when the sequence produces its final token or is canceled

`Flush` is called once per HTTP completion request (not per sequence batch). Since LlamaRunner's `processBatch` serializes under a mutex, there is no concurrent access to the collector from multiple goroutines.

### 5. Activation

```bash
# Disabled (default): NoopCollector, NULL GGML callback, zero overhead
ollama serve

# Enabled: writes one .jsonl per request
OLLAMA_TRACE_DIR=/tmp/traces ollama serve
```

Initialized in `llm/server.go` alongside other `OLLAMA_*` env-var configuration, passed into the runner subprocess via command-line argument or environment variable forwarding.

---

## JSONL Output Format

One file per inference request. A multi-token response produces multiple forward passes.

```jsonl
{"type":"request_start","request_id":"abc123","model":"llama3:8b","n_ctx":4096,"ts":1700000000000}
{"type":"pass_start","pass":0,"n_tokens":512,"ts":1700000001000}
{"type":"op","pass":0,"seq":0,"op":"GET_ROWS","name":"token_embd","srcs":["inp_tokens"],"shape":[512,4096],"dtype":"f16","backend":"CUDA0","t_start":1700000001100,"t_end":1700000001110}
{"type":"op","pass":0,"seq":1,"op":"RMS_NORM","name":"blk.0.attn_norm","srcs":["token_embd"],"shape":[512,4096],"dtype":"f32","backend":"CUDA0","t_start":1700000001110,"t_end":1700000001120}
{"type":"op","pass":0,"seq":2,"op":"MUL_MAT","name":"blk.0.attn_q","srcs":["blk.0.attn_norm","blk.0.attn_q.weight"],"shape":[512,4096],"dtype":"f16","backend":"CUDA0","t_start":1700000001120,"t_end":1700000001350}
{"type":"op","pass":0,"seq":3,"op":"FLASH_ATTN_EXT","name":"blk.0.attn_out","srcs":["blk.0.attn_q","blk.0.attn_k","blk.0.attn_v"],"shape":[512,4096],"dtype":"f16","backend":"CUDA0","t_start":1700000001350,"t_end":1700000001800}
{"type":"op","pass":0,"seq":4,"op":"CPY","name":"blk.0.kv_copy","srcs":["blk.0.attn_k"],"shape":[512,128],"dtype":"f16","backend":"CUDA0","src_backend":"CPU","t_start":1700000001800,"t_end":1700000001950}
{"type":"pass_end","pass":0,"t_start":1700000001100,"t_end":1700000002500,"n_nodes":3847}
{"type":"pass_start","pass":1,"n_tokens":1,"ts":1700000002600}
...
{"type":"request_end","request_id":"abc123","n_passes":128,"prompt_ms":1400,"gen_ms":12300,"ts":1700000015000}
```

### Event Types

| Type | When emitted | Where |
|---|---|---|
| `request_start` | Completion handler begins processing | Go (LlamaRunner) |
| `pass_start` | Before each `lc.Decode()` call | Go (LlamaRunner) |
| `op` | Each GGML compute node dispatched | C callback → Go |
| `pass_end` | After each `lc.Synchronize()` | Go (LlamaRunner) |
| `request_end` | Sequence produces final token or is canceled | Go (LlamaRunner) |

### DAG Reconstruction (post-processing)

```python
nodes = {}  # name → event dict
edges = []  # (src_name, dst_name)

for line in open("trace.jsonl"):
    ev = json.loads(line)
    if ev["type"] == "op":
        nodes[ev["name"]] = ev
        for src in ev["srcs"]:
            edges.append({"from": src, "to": ev["name"],
                          "shape": ev["shape"], "dtype": ev["dtype"]})
```

Copy operations (`op="CPY"`) appear as nodes like any other operator. The visualization can style them differently (dashed edges, different color) to highlight data movement.

### Timing Semantics

`t_start` is recorded on the `ask=true` callback (before dispatch); `t_end` on `ask=false` (after dispatch). For CPU ops, this equals actual execution time. For GPU ops (CUDA/Vulkan), this measures command submission latency — accurate for large ops (matmul, flash-attention where GPU execution dominates); less accurate for small element-wise ops. The precise end-to-end pass time (`pass_end.t_end - pass_start.ts`) provides a ground-truth calibration for relative op costs.

### Op Fusion Note

The GGML graph reflects the *actual fused execution graph*. Flash attention appears as one `FLASH_ATTN_EXT` node; SwiGLU may appear as `SWIGLU`. Intermediate tensors in fused ops do not appear as separate nodes — this is expected and correct; the visualization represents what actually executed. Tensor names encode layer index and component (`blk.N.attn_q`, `blk.N.ffn_gate`), enabling grouping by layer.

---

## Files to Create / Modify

| File | Type | Change |
|------|------|--------|
| `llama/llama.h` | C header (vendored) | Add `llama_context_set_eval_callback()` declaration |
| `llama/llama.cpp` | C++ (vendored) | Implement via `ggml_backend_sched_set_eval_callback()` |
| `llama/llama.go` | Go CGO wrapper | `cgo.Handle` bridge, `//export GoProfilerCallback`, `SetEvalCallback()` method, `profilerHandle` field on `Context` |
| `llm/profiler/profiler.go` | Go (new) | `TraceCollector` interface, `OpEvent`, `New()` factory |
| `llm/profiler/jsonl.go` | Go (new) | `JSONLTraceBuffer` with per-instance mutex, pending map, async `Flush` |
| `llm/profiler/noop.go` | Go (new) | `NoopCollector` |
| `runner/llamarunner/runner.go` | Go | Add `profiler` field to `Server`; inject collector; `RecordPassStart/End`; emit `request_start/end`; `Flush` on completion |
| `llm/server.go` | Go | Read `OLLAMA_TRACE_DIR`, forward to runner subprocess |

---

## Verification Plan

1. Start server: `OLLAMA_TRACE_DIR=/tmp/traces go run . serve`
2. Run 3 requests back-to-back: `ollama run llama3:8b "Hello world"` × 3
3. Confirm 3 `.jsonl` files appear in `/tmp/traces/`
4. Check operator coverage:
   ```bash
   jq -r 'select(.type=="op")|.op' /tmp/traces/trace_*.jsonl | sort | uniq -c | sort -rn
   ```
   Expected: `MUL_MAT`, `RMS_NORM`, `FLASH_ATTN_EXT` (or `SOFT_MAX`+`MUL_MAT`), `GET_ROWS`, `ADD`, `CPY`
5. Check DAG edges: all `srcs` names in `op` events should refer to tensor names that appear elsewhere in the same pass
6. Check GPU assignment: `jq 'select(.type=="op")|.backend' trace.jsonl | sort | uniq -c` — most ops should show `CUDA0` or `Vulkan0`
7. Timing coherence: extract `pass_end.t_end - pass_start.ts` and compare with `processingDuration` from the Sequence struct (should be within same order of magnitude)
8. Zero-overhead check: run without `OLLAMA_TRACE_DIR`; confirm no change in `EvalDuration` in the API response

---

## Development & Testing Workflow

### `OLLAMA_TRACE_DIR` — New Environment Variable (Added by This Feature)

`OLLAMA_TRACE_DIR` does **not** exist in the current ollama codebase. It is introduced by this feature. When unset (the default), a `NoopCollector` is used and a NULL callback is registered in GGML — zero overhead. When set, a `JSONLTraceBuffer` is created and one `.jsonl` file is written per request.

All existing `OLLAMA_*` env vars are documented in `envconfig/config.go`. This new variable should be added there following the same pattern.

---

### Recommended Dev Workflow

**Step 1 — After modifying `llama/llama.h` and `llama/llama.cpp`: confirm C++ compiles**
```bash
go build ./llama/
```
This triggers the CGO compilation of the vendored C++ code. Catches syntax errors and missing symbols immediately.

**Step 2 — After modifying `llama/llama.go` and `llm/profiler/`: run unit tests (no GPU needed, ~seconds)**
```bash
go test ./llama/...
go test ./llm/...
go test ./runner/llamarunner/...
```
These tests cover CGO grammar bindings, KV cache logic, and batch scheduling. They don't test actual inference but catch regressions in the surrounding logic.

> **CGO `//export` gotcha**: The `GoProfilerCallback` function (marked `//export`) must live in a **separate file** (e.g., `llama/profiler_bridge.go`), not in `llama/llama.go`. CGO forbids having both C definitions and `//export` in the same file. Violating this causes a cryptic link error.

**Step 3 — Verify profiling disabled doesn't break inference (integration test, needs GPU + model)**
```bash
# Downloads llama3.2:1b (~1.7 GB) on first run, then runs a real inference
go test -tags=integration -timeout=5m -run TestBlueSky ./integration
```
This is the key regression check. It starts a real ollama server, loads the model, runs a completion, and verifies the response. If this passes with `OLLAMA_TRACE_DIR` unset, the changes are safe for normal usage.

**Step 4 — Verify profiling enabled produces valid output**
```bash
mkdir -p /tmp/traces
OLLAMA_TRACE_DIR=/tmp/traces \
  go test -tags=integration -timeout=5m -run TestBlueSky ./integration

# Inspect the trace
ls /tmp/traces/
cat /tmp/traces/trace_*.jsonl | head -30

# Check operator coverage
cat /tmp/traces/trace_*.jsonl | jq -r 'select(.type=="op")|.op' | sort | uniq -c | sort -rn

# Verify DAG edges (all srcs should be valid tensor names)
cat /tmp/traces/trace_*.jsonl | jq 'select(.type=="op")|{name,srcs}' | head -40

# Check GPU assignment
cat /tmp/traces/trace_*.jsonl | jq -r 'select(.type=="op")|.backend' | sort | uniq -c
```

**Step 5 — Run full unit test suite before committing**
```bash
go test ./...
```

---

### What the Tests Cover vs. What They Don't

| Test | What it verifies | What it doesn't verify |
|---|---|---|
| `go build ./llama/` | C++ compiles, CGO links | Correctness of callback data |
| `go test ./llama/... ./llm/... ./runner/llamarunner/...` | Cache/batch logic, CGO grammar | Actual inference output |
| `TestBlueSky` (no `OLLAMA_TRACE_DIR`) | Inference still works, no crash | Trace data quality |
| `TestBlueSky` (with `OLLAMA_TRACE_DIR`) | Trace files created, no crash | Semantic correctness of all fields |
| Manual `jq` inspection | Operator names, DAG edges, backends | Timing accuracy |

There is no existing test that automatically validates the semantic correctness of trace data (e.g., "all 32 layers appear", "timing values are plausible"). These checks are manual for Phase 1. A dedicated `profiler_test.go` that parses a golden trace file could be added as a follow-on.

---

## Phase 2 Preview (OllamaRunner)

- `ml/backend/ggml/ggml.go`: register the same eval callback on `c.b.sched` (directly accessible in Go — no llama.cpp patch needed for OllamaRunner)
- `runner/ollamarunner/runner.go`: call `profiler.RecordPassStart/End`, `Flush`; emit `request_start/end` events
- Additional benefit: graph structure is built step-by-step in Go before execution, so operator sequence and shapes can also be captured at the Go layer (not only from C callback)
- No changes to the JSONL format or visualization layer required
