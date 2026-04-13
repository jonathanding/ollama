# Native Backend Per-Node Timing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eval callback tracing with backend-native per-node timing (Vulkan GPU timestamps + CPU clock_gettime), achieving <5% overhead vs 2x+ slowdown.

**Architecture:** C layer collects per-node timing inside each backend's graph_compute (Vulkan: vkCmdWriteTimestamp async, CPU: barrier-to-barrier clock). Go layer pulls timing data after sync via new C API, reconstructs absolute timestamps, writes JSONL. Old eval callback mechanism completely removed.

**Tech Stack:** C (ggml-backend, ggml-vulkan, ggml-cpu), Go (CGO bridges, profiler package), Vulkan API

**Spec:** `docs/superpowers/specs/2026-04-13-native-perf-tracing-design.md`

---

## File Structure

### Files to Create

| File | Responsibility |
|------|---------------|
| `llm/profiler/profiler.go` | TraceWriter interface, OpEvent/TensorInfo types, NewWriter factory |
| `llm/profiler/jsonl.go` | JSONLWriter: WriteOps batch, WritePassStart/End, async Flush |
| `llm/profiler/noop.go` | NoopWriter (zero overhead when tracing disabled) |
| `llm/profiler/profiler_test.go` | Unit tests for TraceWriter |
| `ml/backend/ggml/timing_bridge.go` | OllamaRunner CGO bridge: EnableTiming + CollectTiming |
| `llama/timing_bridge.go` | LlamaRunner CGO bridge: EnableTiming + CollectTiming |

### Files to Modify

| File | Change |
|------|--------|
| `ml/backend/ggml/ggml/src/ggml-backend-impl.h:138-143` | Add `bool timing_enabled` to `ggml_backend` struct |
| `ml/backend/ggml/ggml/include/ggml-backend.h:304-354` | Remove eval callback type/decl, add timing API |
| `ml/backend/ggml/ggml/src/ggml-backend.cpp:712-760,1616-1653,1897` | Remove callback fields/branch/function, add timing API impl |
| `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp:99-108` | Add timing fields to cpu_context |
| `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c:2940-2963` | Add barrier-to-barrier timing on thread 0 |
| `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp:1678-1737` | Add node_timing_ns vector, collect_timing function |
| `llama/llama.cpp/include/llama.h:948` | Replace set_eval_callback with timing wrappers |
| `llama/llama.cpp/src/llama-context.cpp:2529-2536` | Replace set_eval_callback with timing impl |
| `runner/ollamarunner/runner.go:332-391,642,1272-1277` | Remove old prof, add traceWriter + timing pull |
| `runner/llamarunner/runner.go:257-311,412,867` | Remove old prof, add traceWriter + timing pull |
| `ml/backend/ggml/ggml.go:122,464` | Remove profilerHandle field + cleanup |
| `llama/llama.go:164` | Remove profilerHandle field |

### Files to Delete

| File | Reason |
|------|--------|
| `llm/profiler/profiler.go` | Old TraceCollector interface (replaced) |
| `llm/profiler/jsonl.go` | Old JSONLTraceBuffer (replaced) |
| `llm/profiler/noop.go` | Old NoopCollector (replaced) |
| `llm/profiler/profiler_test.go` | Old tests (replaced) |
| `llama/profiler_bridge.go` | Old LlamaRunner eval callback CGO bridge |
| `ml/backend/ggml/profiler_bridge.go` | Old OllamaRunner eval callback CGO bridge |

---

## Task 1: New Go Profiler Package (TDD)

**Files:**
- Delete: `llm/profiler/profiler.go`, `llm/profiler/jsonl.go`, `llm/profiler/noop.go`, `llm/profiler/profiler_test.go`
- Create: `llm/profiler/profiler.go`, `llm/profiler/jsonl.go`, `llm/profiler/noop.go`, `llm/profiler/profiler_test.go`

- [ ] **Step 1: Delete old profiler package files**

```bash
rm llm/profiler/profiler.go llm/profiler/jsonl.go llm/profiler/noop.go llm/profiler/profiler_test.go
```

Note: This breaks the build for packages that import `llm/profiler`. That's expected — we'll fix it in Tasks 8-9.

- [ ] **Step 2: Write the test file**

Create `llm/profiler/profiler_test.go`:

```go
package profiler

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNoopWriter(t *testing.T) {
	w := NewWriter("")
	if _, ok := w.(*NoopWriter); !ok {
		t.Fatalf("expected NoopWriter, got %T", w)
	}
	w.WritePassStart(0, 32)
	w.WriteOps([]OpEvent{{Type: "op", Op: "MUL_MAT"}})
	w.WritePassEnd(0)
	if err := w.Flush("req1", "model1"); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestJSONLWriterFlush(t *testing.T) {
	dir := t.TempDir()
	w := NewWriter(dir)
	jw, ok := w.(*JSONLWriter)
	if !ok {
		t.Fatalf("expected JSONLWriter, got %T", w)
	}
	_ = jw

	w.WritePassStart(0, 4)
	w.WriteOps([]OpEvent{
		{Type: "op", Op: "MUL_MAT", Name: "blk.0.attn_q.weight", Backend: "Vulkan",
			SrcNames: []string{"src0", "src1"}, OutShape: []int64{4096, 32, 1, 1}, DType: "f16",
			TStart: 1000, TEnd: 5000},
		{Type: "op", Op: "ADD", Name: "blk.0.attn_q.bias", Backend: "CPU",
			SrcNames: []string{"src0"}, OutShape: []int64{4096, 32, 1, 1}, DType: "f32",
			TStart: 5000, TEnd: 5500},
	})
	w.WritePassEnd(0)

	err := w.Flush("test-req", "testmodel")
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)

	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("expected 1 trace file, got %d", len(entries))
	}

	data, _ := os.ReadFile(filepath.Join(dir, entries[0].Name()))
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 JSONL lines, got %d", len(lines))
	}

	var passStart map[string]any
	json.Unmarshal([]byte(lines[0]), &passStart)
	if passStart["type"] != "pass_start" {
		t.Errorf("line 0: expected pass_start, got %v", passStart["type"])
	}

	var op OpEvent
	json.Unmarshal([]byte(lines[1]), &op)
	if op.Type != "op" || op.Op != "MUL_MAT" || op.SeqID != 0 {
		t.Errorf("line 1: unexpected op %+v", op)
	}
	var op2 OpEvent
	json.Unmarshal([]byte(lines[2]), &op2)
	if op2.SeqID != 1 {
		t.Errorf("line 2: expected SeqID=1, got %d", op2.SeqID)
	}

	var passEnd map[string]any
	json.Unmarshal([]byte(lines[3]), &passEnd)
	if passEnd["type"] != "pass_end" {
		t.Errorf("line 3: expected pass_end, got %v", passEnd["type"])
	}
}

func TestJSONLWriterMultiPass(t *testing.T) {
	dir := t.TempDir()
	w := NewWriter(dir)

	w.WritePassStart(0, 4)
	w.WriteOps([]OpEvent{{Type: "op", Op: "MUL_MAT", TStart: 100, TEnd: 200}})
	w.WritePassEnd(0)
	w.WritePassStart(1, 8)
	w.WriteOps([]OpEvent{{Type: "op", Op: "ADD", TStart: 300, TEnd: 350}})
	w.WritePassEnd(1)

	w.Flush("multi-pass", "testmodel")
	time.Sleep(100 * time.Millisecond)

	entries, _ := os.ReadDir(dir)
	data, _ := os.ReadFile(filepath.Join(dir, entries[0].Name()))
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 6 {
		t.Fatalf("expected 6 JSONL lines, got %d", len(lines))
	}

	var op OpEvent
	json.Unmarshal([]byte(lines[4]), &op)
	if op.SeqID != 0 {
		t.Errorf("pass 1 op: expected SeqID=0, got %d", op.SeqID)
	}
}

func TestFlushFilenameFormat(t *testing.T) {
	dir := t.TempDir()
	w := NewWriter(dir)
	w.WriteOps([]OpEvent{{Type: "op", Op: "NOP", TStart: 0, TEnd: 1}})
	w.Flush("req/with special chars!", "model")
	time.Sleep(100 * time.Millisecond)

	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("expected 1 file, got %d", len(entries))
	}
	name := entries[0].Name()
	if !strings.HasPrefix(name, "trace_req_with_special_chars_") {
		t.Errorf("unexpected filename: %s", name)
	}
	if !strings.HasSuffix(name, ".jsonl") {
		t.Errorf("expected .jsonl suffix: %s", name)
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `go test ./llm/profiler/ -v`
Expected: FAIL — package has no source files

- [ ] **Step 4: Write profiler.go**

Create `llm/profiler/profiler.go`:

```go
package profiler

// TraceWriter is the pull-model tracing interface.
// Go pulls timing data from C backends after sync, then calls WriteOps.
type TraceWriter interface {
	WriteOps(ops []OpEvent)
	WritePassStart(passID int, nTokens int)
	WritePassEnd(passID int)
	Flush(requestID string, model string) error
	Close() error
}

// OpEvent represents one GGML operator node execution.
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

// NewWriter returns a JSONLWriter if outDir is non-empty, otherwise NoopWriter.
func NewWriter(outDir string) TraceWriter {
	if outDir == "" {
		return &NoopWriter{}
	}
	return newJSONLWriter(outDir)
}
```

- [ ] **Step 5: Write noop.go**

Create `llm/profiler/noop.go`:

```go
package profiler

// NoopWriter implements TraceWriter with zero overhead.
type NoopWriter struct{}

func (n *NoopWriter) WriteOps(_ []OpEvent)           {}
func (n *NoopWriter) WritePassStart(_ int, _ int)     {}
func (n *NoopWriter) WritePassEnd(_ int)              {}
func (n *NoopWriter) Flush(_ string, _ string) error  { return nil }
func (n *NoopWriter) Close() error                    { return nil }
```

- [ ] **Step 6: Write jsonl.go**

Create `llm/profiler/jsonl.go`:

```go
package profiler

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type JSONLWriter struct {
	mu     sync.Mutex
	lines  [][]byte
	outDir string
	seqID  int
}

func newJSONLWriter(outDir string) *JSONLWriter {
	return &JSONLWriter{outDir: outDir}
}

func (w *JSONLWriter) WriteOps(ops []OpEvent) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for i := range ops {
		ops[i].SeqID = w.seqID
		w.seqID++
		line, _ := json.Marshal(ops[i])
		w.lines = append(w.lines, line)
	}
}

func (w *JSONLWriter) WritePassStart(passID int, nTokens int) {
	line, _ := json.Marshal(map[string]any{
		"type": "pass_start", "pass": passID, "n_tokens": nTokens,
		"ts": time.Now().UnixMilli(),
	})
	w.mu.Lock()
	w.lines = append(w.lines, line)
	w.seqID = 0
	w.mu.Unlock()
}

func (w *JSONLWriter) WritePassEnd(passID int) {
	line, _ := json.Marshal(map[string]any{
		"type": "pass_end", "pass": passID,
		"ts": time.Now().UnixMilli(),
	})
	w.mu.Lock()
	w.lines = append(w.lines, line)
	w.mu.Unlock()
}

func (w *JSONLWriter) Flush(requestID, model string) error {
	w.mu.Lock()
	lines := w.lines
	w.lines = nil
	w.mu.Unlock()

	if len(lines) == 0 {
		return nil
	}

	go func() {
		ts := time.Now().UnixMilli()
		safe := strings.Map(func(r rune) rune {
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
				(r >= '0' && r <= '9') || r == '-' || r == '_' {
				return r
			}
			return '_'
		}, requestID)
		fname := filepath.Join(w.outDir, fmt.Sprintf("trace_%s_%d.jsonl", safe, ts))
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

func (w *JSONLWriter) Close() error { return nil }
```

- [ ] **Step 7: Run tests**

Run: `go test ./llm/profiler/ -v`
Expected: PASS — all 4 tests green

- [ ] **Step 8: Commit**

```bash
git add llm/profiler/
git commit -m "profiler: rewrite package with pull-model TraceWriter interface"
```

---

## Task 2: C Layer — ggml-backend Timing API

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-backend-impl.h:138-143`
- Modify: `ml/backend/ggml/ggml/include/ggml-backend.h:304-354`
- Modify: `ml/backend/ggml/ggml/src/ggml-backend.cpp:712-760,1616-1653,1897`

- [ ] **Step 1: Add `timing_enabled` to ggml_backend struct**

In `ml/backend/ggml/ggml/src/ggml-backend-impl.h`, add field to `ggml_backend` struct (line 138):

```c
    struct ggml_backend {
        ggml_guid_t guid;
        struct ggml_backend_i iface;
        ggml_backend_dev_t device;
        void * context;
        bool timing_enabled;
    };
```

- [ ] **Step 2: Remove old callback declarations from ggml-backend.h**

In `ml/backend/ggml/ggml/include/ggml-backend.h`:

Delete the eval callback typedef and comment block (lines 304-311):
```c
// DELETE: typedef bool (*ggml_backend_sched_eval_callback)(...);
```

Delete the set_eval_callback declaration (line 354):
```c
// DELETE: GGML_API void ggml_backend_sched_set_eval_callback(...);
```

- [ ] **Step 3: Add new timing API declarations to ggml-backend.h**

In `ml/backend/ggml/ggml/include/ggml-backend.h`, after the `ggml_backend_sched_reset` declaration (line 351), add:

```c
    // Native per-node timing API
    GGML_API void ggml_backend_set_timing(ggml_backend_t backend, bool enabled);
    GGML_API void ggml_backend_sched_set_timing(ggml_backend_sched_t sched, bool enabled);

    // Split query (read-only, valid between graph_compute and next graph_compute)
    GGML_API int  ggml_backend_sched_get_split_start(ggml_backend_sched_t sched, int split_id);
    GGML_API int  ggml_backend_sched_get_split_n_nodes(ggml_backend_sched_t sched, int split_id);
    GGML_API int  ggml_backend_sched_get_split_backend_id(ggml_backend_sched_t sched, int split_id);

    // Read timing data for a split (call after synchronize, before next graph_compute)
    GGML_API int  ggml_backend_sched_get_split_timing(
                      ggml_backend_sched_t sched, int split_id,
                      uint64_t * timing_out, int capacity);

    // Node metadata extraction
    struct ggml_node_info {
        const char * op_name;
        const char * tensor_name;
        int64_t      shape[4];
        const char * dtype_name;
        const char * backend_name;
        const char * src_names[GGML_MAX_SRC];
        int          n_srcs;
    };

    GGML_API void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out);
```

Note: `ggml_backend_sched_get_n_splits` already exists at line 329 — don't re-declare it.

- [ ] **Step 4: Remove callback fields from ggml_backend_sched struct**

In `ml/backend/ggml/ggml/src/ggml-backend.cpp`, in the `ggml_backend_sched` struct (line 712), delete lines 751-752:

```c
// DELETE these two lines:
    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;
```

Add new field (anywhere in the struct, e.g., after line 757):

```c
    bool timing_enabled;
```

- [ ] **Step 5: Delete callback branch in compute_splits**

In `ml/backend/ggml/ggml/src/ggml-backend.cpp`, replace the if/else block at lines 1616-1653:

Before:
```c
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph, sched->batch_size);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // ... 30+ lines of callback logic ...
        }
```

After:
```c
        {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph, sched->batch_size);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        }
```

- [ ] **Step 6: Delete ggml_backend_sched_set_eval_callback function**

In `ml/backend/ggml/ggml/src/ggml-backend.cpp`, delete the function at line 1897:

```c
// DELETE:
void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}
```

- [ ] **Step 7: Implement timing API functions**

In `ml/backend/ggml/ggml/src/ggml-backend.cpp`, add after the `ggml_backend_sched_synchronize` function (line 1884):

```c
// --- Native per-node timing API ---

void ggml_backend_set_timing(ggml_backend_t backend, bool enabled) {
    backend->timing_enabled = enabled;
}

void ggml_backend_sched_set_timing(ggml_backend_sched_t sched, bool enabled) {
    sched->timing_enabled = enabled;
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_set_timing(sched->backends[i], enabled);
    }
}

int ggml_backend_sched_get_split_start(ggml_backend_sched_t sched, int split_id) {
    GGML_ASSERT(split_id >= 0 && split_id < sched->n_splits);
    return sched->splits[split_id].i_start;
}

int ggml_backend_sched_get_split_n_nodes(ggml_backend_sched_t sched, int split_id) {
    GGML_ASSERT(split_id >= 0 && split_id < sched->n_splits);
    return sched->splits[split_id].graph.n_nodes;
}

int ggml_backend_sched_get_split_backend_id(ggml_backend_sched_t sched, int split_id) {
    GGML_ASSERT(split_id >= 0 && split_id < sched->n_splits);
    return sched->splits[split_id].backend_id;
}

int ggml_backend_sched_get_split_timing(
        ggml_backend_sched_t sched, int split_id,
        uint64_t * timing_out, int capacity) {
    GGML_ASSERT(split_id >= 0 && split_id < sched->n_splits);
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    ggml_backend_t backend = sched->backends[split->backend_id];

    if (!backend->timing_enabled) {
        memset(timing_out, 0, capacity * sizeof(uint64_t));
        return 0;
    }

    // Dispatch to backend-specific timing collection.
    // Each backend stores timing in its own context.
    // The backend's collect_timing function fills timing_out with per-node
    // elapsed nanoseconds for nodes in this split's sub-graph.
    //
    // For backends that don't support timing (e.g., CUDA), this returns 0
    // and timing_out is zeroed.
    int n_nodes = split->graph.n_nodes;
    int count = n_nodes < capacity ? n_nodes : capacity;

    // Try backend-specific collection via name dispatch
    const char * name = ggml_backend_name(backend);
    if (strncmp(name, "Vulkan", 6) == 0) {
        extern int ggml_vk_collect_timing(ggml_backend_t backend, uint64_t * timing_out, int capacity);
        return ggml_vk_collect_timing(backend, timing_out, count);
    }
    if (strcmp(name, "CPU") == 0) {
        extern int ggml_cpu_collect_timing(ggml_backend_t backend, uint64_t * timing_out, int capacity);
        return ggml_cpu_collect_timing(backend, timing_out, count);
    }

    // Unsupported backend — return zeros
    memset(timing_out, 0, count * sizeof(uint64_t));
    return 0;
}

void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out) {
    out->op_name     = ggml_op_name(node->op);
    out->tensor_name = node->name;
    out->dtype_name  = ggml_type_name(node->type);

    // Backend name from buffer
    if (node->buffer) {
        out->backend_name = ggml_backend_buffer_name(node->buffer);
    } else {
        out->backend_name = "unknown";
    }

    for (int i = 0; i < 4; i++) {
        out->shape[i] = node->ne[i];
    }

    out->n_srcs = 0;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i]) {
            out->src_names[i] = node->src[i]->name;
            out->n_srcs = i + 1;
        } else {
            out->src_names[i] = NULL;
        }
    }
}
```

- [ ] **Step 8: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-backend-impl.h \
        ml/backend/ggml/ggml/include/ggml-backend.h \
        ml/backend/ggml/ggml/src/ggml-backend.cpp
git commit -m "ggml-backend: add native timing API, remove eval callback"
```

---

## Task 3: C Layer — CPU Backend Timing

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp:99-108,116-121,167`
- Modify: `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c:2940-2963`

- [ ] **Step 1: Add timing fields to ggml_backend_cpu_context**

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp`, add two fields to the struct (line 99):

```c
struct ggml_backend_cpu_context {
    int                 n_threads;
    ggml_threadpool_t   threadpool;

    uint8_t *           work_data;
    size_t              work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // Per-node timing (populated when backend->timing_enabled)
    uint64_t *          node_timing_ns;
    int                 timing_capacity;
};
```

- [ ] **Step 2: Add cleanup in ggml_backend_cpu_free**

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp`, in `ggml_backend_cpu_free` (line 116), add before `delete cpu_ctx`:

```c
static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    delete[] cpu_ctx->work_data;
    delete[] cpu_ctx->node_timing_ns;
    delete cpu_ctx;
    delete backend;
}
```

- [ ] **Step 3: Add timing realloc in ggml_backend_cpu_graph_compute**

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp`, in `ggml_backend_cpu_graph_compute` (line 167), add timing capacity check near the top of the function, after the cpu_ctx cast:

```c
static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph, int batch_size) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    // Ensure timing array capacity
    if (backend->timing_enabled && cgraph->n_nodes > cpu_ctx->timing_capacity) {
        delete[] cpu_ctx->node_timing_ns;
        cpu_ctx->node_timing_ns = new uint64_t[cgraph->n_nodes]();
        cpu_ctx->timing_capacity = cgraph->n_nodes;
    }

    // ... rest of function unchanged ...
```

- [ ] **Step 4: Add barrier-to-barrier timing in ggml_graph_compute_thread**

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c`, modify the per-node loop (line 2940). This file is C, not C++.

The function needs access to backend->timing_enabled and cpu_ctx. The thread function receives `struct ggml_compute_state*` which has `threadpool`, which has the `cgraph` and `cplan`. We need to pass the backend pointer or timing array through the existing structures.

**Approach:** Pass the timing array pointer through `cplan->work_data` area or add a field to `ggml_cplan`. Simplest: store timing pointer in the threadpool struct. However, threadpool is a ggml-internal struct.

**Better approach:** Store the timing array pointer and enabled flag directly as file-scope variables set by `ggml_backend_cpu_graph_compute` before calling `ggml_graph_compute`. This avoids modifying shared structs:

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c`, add file-scope variables near the top:

```c
// Per-compute timing state (set by graph_compute, read by compute_thread)
static _Thread_local uint64_t * ggml_cpu_timing_ns = NULL;
```

Wait — this won't work because multiple threads read the same pointer. Better: use a regular static set before the threadpool compute call:

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp`, in `ggml_backend_cpu_graph_compute`, **before** calling `ggml_graph_compute`:

```c
    // Pass timing array to the compute thread via extern
    extern void ggml_cpu_set_timing_state(uint64_t * timing_ns);
    if (backend->timing_enabled) {
        memset(cpu_ctx->node_timing_ns, 0, cgraph->n_nodes * sizeof(uint64_t));
        ggml_cpu_set_timing_state(cpu_ctx->node_timing_ns);
    } else {
        ggml_cpu_set_timing_state(NULL);
    }
```

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c`, add near the top (after includes):

```c
// Timing state set by ggml_backend_cpu_graph_compute, read by thread 0
static uint64_t * ggml_cpu_node_timing_ns = NULL;

void ggml_cpu_set_timing_state(uint64_t * timing_ns) {
    ggml_cpu_node_timing_ns = timing_ns;
}
```

Then modify the per-node loop in `ggml_graph_compute_thread` (line 2940):

```c
    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            continue;
        }

        uint64_t t0 = 0;
        if (state->ith == 0 && ggml_cpu_node_timing_ns) {
            t0 = ggml_time_us();
        }

        ggml_compute_forward(&params, node);

#ifdef OLLAMA_DEBUG
        ollama_debug(node, true);
#endif

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }

        if (state->ith == 0 && t0 != 0) {
            ggml_cpu_node_timing_ns[node_n] = (ggml_time_us() - t0) * 1000;
        }
    }
```

- [ ] **Step 5: Add ggml_cpu_collect_timing function**

In `ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.cpp`, add after `ggml_backend_cpu_graph_compute`:

```c
// Called by ggml_backend_sched_get_split_timing via name dispatch
int ggml_cpu_collect_timing(ggml_backend_t backend, uint64_t * timing_out, int capacity) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    if (!cpu_ctx->node_timing_ns) {
        memset(timing_out, 0, capacity * sizeof(uint64_t));
        return 0;
    }
    int count = capacity < cpu_ctx->timing_capacity ? capacity : cpu_ctx->timing_capacity;
    memcpy(timing_out, cpu_ctx->node_timing_ns, count * sizeof(uint64_t));
    return count;
}
```

- [ ] **Step 6: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-cpu/
git commit -m "ggml-cpu: add per-node barrier-to-barrier timing on thread 0"
```

---

## Task 4: C Layer — Vulkan Backend Timing

**Files:**
- Modify: `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp:1678-1737,13156-13353`

The existing perf logger code (gated on `vk_perf_logger_enabled`) already inserts `vkCmdWriteTimestamp` and reads query pool results. Our timing uses the **same query pool infrastructure** but with a critical difference: timestamps are inserted async (no `waitForFences` inside `graph_compute`), and results are read **later** by `ggml_vk_collect_timing()` (called after Go-side sync).

- [ ] **Step 1: Add node_timing_ns to vk_context**

In `ml/backend/ggml/ggml/src/ggml-vulkan/ggml-vulkan.cpp`, add to `ggml_backend_vk_context` struct (line 1678), after existing `query_idx` field (line 1736):

```c
    // Native timing (coexists with perf_logger)
    std::vector<uint64_t> node_timing_ns;
    int timing_query_count {};  // number of timestamps inserted this compute
```

- [ ] **Step 2: Gate timestamp insertion on timing_enabled**

In the `graph_compute` function, the existing perf logger code at line 13156 gates on `vk_perf_logger_enabled`. We need to also execute the same query pool setup and timestamp insertion when `backend->timing_enabled` is true.

Modify the condition at line 13156 and the per-node insertion at line 13294:

At line 13156, change:
```c
    if (vk_perf_logger_enabled) {
```
to:
```c
    bool need_timestamps = vk_perf_logger_enabled || backend->timing_enabled;
    if (need_timestamps) {
```

At line 13294, change:
```c
        if (vk_perf_logger_enabled && enqueued) {
```
to:
```c
        if (need_timestamps && enqueued) {
```

Note: The `backend` pointer is the first parameter of `ggml_backend_vk_graph_compute`. Ensure `need_timestamps` is visible throughout the function (declare at function scope).

- [ ] **Step 3: Add timing array resize alongside query pool resize**

Inside the query pool allocation block (line 13158), after `ctx->query_nodes.resize(...)` at line 13168, add:

```c
            ctx->node_timing_ns.resize(ctx->num_queries);
```

- [ ] **Step 4: Separate perf_logger processing from timestamp insertion**

The perf logger's `waitForFences` + `getQueryPoolResults` block at lines 13333-13353 must NOT run when only `backend->timing_enabled` is true (it would make compute synchronous). Modify:

At line 13333, change:
```c
    if (vk_perf_logger_enabled) {
```
to:
```c
    if (vk_perf_logger_enabled && !backend->timing_enabled) {
        // Perf logger uses synchronous read — only when native timing is NOT enabled
        // (native timing reads asynchronously after Go-side sync)
```

This means when both are enabled, native timing takes precedence (async). The perf logger's stderr output will be disabled in that case. If only `vk_perf_logger_enabled`, behavior is unchanged.

If only `backend->timing_enabled` (not perf_logger), we need to ensure the command buffer is still properly ended and submitted. Add after the modified perf_logger block:

```c
    if (backend->timing_enabled && !vk_perf_logger_enabled) {
        // Ensure any pending compute context is submitted (timestamps included in command buffer)
        if (!ctx->compute_ctx.expired()) {
            compute_ctx = ctx->compute_ctx.lock();
            ggml_vk_ctx_end(compute_ctx);
            ggml_vk_submit(compute_ctx, ctx->device->fence);
        }
        ctx->timing_query_count = ctx->query_idx;
    }
```

Also store timing_query_count in the perf_logger path too:
```c
    if (need_timestamps) {
        ctx->timing_query_count = ctx->query_idx;
    }
```

- [ ] **Step 5: Implement ggml_vk_collect_timing**

Add after `ggml_backend_vk_graph_compute` (near end of file):

```cpp
// Called by ggml_backend_sched_get_split_timing after Go-side sync.
// At this point, all fences are signaled and query pool results are valid.
extern "C" int ggml_vk_collect_timing(ggml_backend_t backend, uint64_t * timing_out, int capacity) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    if (ctx->timing_query_count < 2 || !ctx->query_pool) {
        memset(timing_out, 0, capacity * sizeof(uint64_t));
        return 0;
    }

    // Read timestamps from query pool (fences already signaled)
    std::vector<uint64_t> timestamps(ctx->timing_query_count);
    VK_CHECK(ctx->device->device.getQueryPoolResults(
        ctx->query_pool, 0, ctx->timing_query_count,
        ctx->timing_query_count * sizeof(uint64_t),
        timestamps.data(), sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait),
        "collect_timing getQueryPoolResults");

    double period = ctx->device->properties.limits.timestampPeriod;
    int count = (ctx->timing_query_count - 1) < capacity ? (ctx->timing_query_count - 1) : capacity;

    for (int i = 0; i < count; i++) {
        timing_out[i] = (uint64_t)((timestamps[i + 1] - timestamps[i]) * period);
    }

    return count;
}
```

- [ ] **Step 6: Commit**

```bash
git add ml/backend/ggml/ggml/src/ggml-vulkan/
git commit -m "ggml-vulkan: add async timestamp collection for native timing"
```

---

## Task 5: C Layer — llama.cpp API Wrappers

**Files:**
- Modify: `llama/llama.cpp/include/llama.h:948`
- Modify: `llama/llama.cpp/src/llama-context.cpp:2529-2536`
- Modify: `llama/llama.cpp/src/llama-context.h` (member declarations)

- [ ] **Step 1: Replace set_eval_callback declaration in llama.h**

In `llama/llama.cpp/include/llama.h`, replace the `llama_context_set_eval_callback` declaration (line 948) with timing wrapper declarations:

```c
    // Native per-node timing (replaces eval callback)
    LLAMA_API void llama_context_enable_timing(struct llama_context * ctx, bool enabled);
    LLAMA_API int  llama_context_get_n_splits(struct llama_context * ctx);
    LLAMA_API int  llama_context_get_split_start(struct llama_context * ctx, int split_id);
    LLAMA_API int  llama_context_get_split_n_nodes(struct llama_context * ctx, int split_id);
    LLAMA_API int  llama_context_get_split_backend_id(struct llama_context * ctx, int split_id);
    LLAMA_API int  llama_context_get_split_timing(struct llama_context * ctx, int split_id,
                                                   uint64_t * out, int capacity);
    LLAMA_API struct ggml_cgraph * llama_context_get_graph(struct llama_context * ctx);
```

Also remove any `ggml_backend_sched_eval_callback` usage in `llama_context_params` (llama.h:344 — `cb_eval` and `cb_eval_user_data` fields). Set these to `// removed` or delete them entirely.

- [ ] **Step 2: Replace set_eval_callback implementation in llama-context.cpp**

In `llama/llama.cpp/src/llama-context.cpp`, replace the old implementation (lines 2529-2536) with:

```c
void llama_context_enable_timing(struct llama_context * ctx, bool enabled) {
    ggml_backend_sched_set_timing(ctx->sched.get(), enabled);
}

int llama_context_get_n_splits(struct llama_context * ctx) {
    return ggml_backend_sched_get_n_splits(ctx->sched.get());
}

int llama_context_get_split_start(struct llama_context * ctx, int split_id) {
    return ggml_backend_sched_get_split_start(ctx->sched.get(), split_id);
}

int llama_context_get_split_n_nodes(struct llama_context * ctx, int split_id) {
    return ggml_backend_sched_get_split_n_nodes(ctx->sched.get(), split_id);
}

int llama_context_get_split_backend_id(struct llama_context * ctx, int split_id) {
    return ggml_backend_sched_get_split_backend_id(ctx->sched.get(), split_id);
}

int llama_context_get_split_timing(struct llama_context * ctx, int split_id,
                                    uint64_t * out, int capacity) {
    return ggml_backend_sched_get_split_timing(ctx->sched.get(), split_id, out, capacity);
}

struct ggml_cgraph * llama_context_get_graph(struct llama_context * ctx) {
    return ctx->gf;
}
```

Note: `ctx->sched` is a `std::unique_ptr<ggml_backend_sched>` — use `.get()`. `ctx->gf` is the current computation graph pointer. Verify the field name by searching for `ggml_cgraph` in `llama-context.h`.

- [ ] **Step 3: Remove cb_eval usage in context initialization**

In `llama/llama.cpp/src/llama-context.cpp`, find where `cb_eval` from params is used (line ~831) and remove the `ggml_backend_sched_set_eval_callback` call. Also remove the member function declaration in `llama-context.h`.

- [ ] **Step 4: Commit**

```bash
git add llama/llama.cpp/include/llama.h \
        llama/llama.cpp/src/llama-context.cpp \
        llama/llama.cpp/src/llama-context.h \
        llama/llama.cpp/src/llama-cparams.h
git commit -m "llama: replace set_eval_callback with native timing wrappers"
```

---

## Task 6: CGO Bridge — OllamaRunner

**Files:**
- Create: `ml/backend/ggml/timing_bridge.go`
- Delete: `ml/backend/ggml/profiler_bridge.go`

- [ ] **Step 1: Delete old profiler bridge**

```bash
rm ml/backend/ggml/profiler_bridge.go
```

- [ ] **Step 2: Create timing_bridge.go**

Create `ml/backend/ggml/timing_bridge.go`:

```go
package ggml

/*
#include "ggml-backend.h"

extern void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out);
*/
import "C"

import (
	"time"
	"unsafe"

	"github.com/ollama/ollama/llm/profiler"
)

// EnableTiming enables/disables per-node timing collection on all backends.
func (b *Backend) EnableTiming(enabled bool) {
	C.ggml_backend_sched_set_timing(b.sched, C.bool(enabled))
}

// CollectTiming reads per-node timing data from all splits after sync.
// Must be called after synchronize and before next graph_compute.
func (b *Backend) CollectTiming(passStartTime time.Time) []profiler.OpEvent {
	nSplits := int(C.ggml_backend_sched_get_n_splits(b.sched))
	var events []profiler.OpEvent

	for s := 0; s < nSplits; s++ {
		nNodes := int(C.ggml_backend_sched_get_split_n_nodes(b.sched, C.int(s)))
		if nNodes == 0 {
			continue
		}
		iStart := int(C.ggml_backend_sched_get_split_start(b.sched, C.int(s)))

		buf := make([]uint64, nNodes)
		n := int(C.ggml_backend_sched_get_split_timing(
			b.sched, C.int(s),
			(*C.uint64_t)(unsafe.Pointer(&buf[0])), C.int(nNodes)))
		if n == 0 {
			continue
		}

		for j := 0; j < n; j++ {
			elapsed := buf[j]
			if elapsed == 0 {
				continue
			}

			// Get the node from the full graph via split offset
			node := b.lastGraphNodeAt(iStart + j)
			if node == nil {
				continue
			}

			var info C.struct_ggml_node_info
			C.ggml_node_get_info(node, &info)

			ev := profiler.OpEvent{
				Type:    "op",
				Op:      C.GoString(info.op_name),
				Name:    C.GoString(info.tensor_name),
				DType:   C.GoString(info.dtype_name),
				Backend: C.GoString(info.backend_name),
				TEnd:    int64(elapsed), // temporary: holds duration, reconstructed below
			}

			// Extract shape
			for d := 0; d < 4; d++ {
				ev.OutShape = append(ev.OutShape, int64(info.shape[d]))
			}

			// Extract source names
			for k := 0; k < int(info.n_srcs); k++ {
				if info.src_names[k] != nil {
					ev.SrcNames = append(ev.SrcNames, C.GoString(info.src_names[k]))
				}
			}

			events = append(events, ev)
		}
	}

	// Reconstruct absolute timestamps from cumulative elapsed
	cursor := passStartTime.UnixNano()
	for i := range events {
		events[i].TStart = cursor
		events[i].TEnd = cursor + events[i].TEnd
		cursor = events[i].TEnd
	}

	return events
}

// graphNodeAt returns the tensor node at the given index in the last computed graph.
func (b *Backend) graphNodeAt(idx int) *C.struct_ggml_tensor {
	// b.lastGraph is the ggml_cgraph from the last ComputeWithNotify call.
	// Access: b.lastGraph.nodes[idx]
	if b.lastGraph == nil || C.int(idx) >= b.lastGraph.n_nodes {
		return nil
	}
	nodes := unsafe.Slice(b.lastGraph.nodes, b.lastGraph.n_nodes)
	return nodes[idx]
}
```

Note: The `b.lastGraph` field needs to be set during `ComputeWithNotify` — see Task 8 for the modification to `ggml.go`.

- [ ] **Step 3: Commit**

```bash
git add ml/backend/ggml/timing_bridge.go
git rm ml/backend/ggml/profiler_bridge.go
git commit -m "ggml: add timing CGO bridge for OllamaRunner"
```

---

## Task 7: CGO Bridge — LlamaRunner

**Files:**
- Create: `llama/timing_bridge.go`
- Delete: `llama/profiler_bridge.go`

- [ ] **Step 1: Delete old profiler bridge**

```bash
rm llama/profiler_bridge.go
```

- [ ] **Step 2: Create timing_bridge.go**

Create `llama/timing_bridge.go`:

```go
package llama

/*
#include "llama.h"
#include "ggml-backend.h"

extern void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out);
*/
import "C"

import (
	"time"
	"unsafe"

	"github.com/ollama/ollama/llm/profiler"
)

// EnableTiming enables/disables per-node timing on this llama context.
func (c *Context) EnableTiming(enabled bool) {
	C.llama_context_enable_timing(c.c, C.bool(enabled))
}

// CollectTiming reads per-node timing data after Synchronize.
func (c *Context) CollectTiming(passStartTime time.Time) []profiler.OpEvent {
	nSplits := int(C.llama_context_get_n_splits(c.c))
	var events []profiler.OpEvent

	graph := C.llama_context_get_graph(c.c)
	if graph == nil {
		return nil
	}

	for s := 0; s < nSplits; s++ {
		nNodes := int(C.llama_context_get_split_n_nodes(c.c, C.int(s)))
		if nNodes == 0 {
			continue
		}
		iStart := int(C.llama_context_get_split_start(c.c, C.int(s)))

		buf := make([]uint64, nNodes)
		n := int(C.llama_context_get_split_timing(
			c.c, C.int(s),
			(*C.uint64_t)(unsafe.Pointer(&buf[0])), C.int(nNodes)))
		if n == 0 {
			continue
		}

		for j := 0; j < n; j++ {
			elapsed := buf[j]
			if elapsed == 0 {
				continue
			}

			nodeIdx := iStart + j
			if C.int(nodeIdx) >= graph.n_nodes {
				continue
			}
			nodes := unsafe.Slice(graph.nodes, graph.n_nodes)
			node := nodes[nodeIdx]

			var info C.struct_ggml_node_info
			C.ggml_node_get_info(node, &info)

			ev := profiler.OpEvent{
				Type:    "op",
				Op:      C.GoString(info.op_name),
				Name:    C.GoString(info.tensor_name),
				DType:   C.GoString(info.dtype_name),
				Backend: C.GoString(info.backend_name),
				TEnd:    int64(elapsed),
			}

			for d := 0; d < 4; d++ {
				ev.OutShape = append(ev.OutShape, int64(info.shape[d]))
			}
			for k := 0; k < int(info.n_srcs); k++ {
				if info.src_names[k] != nil {
					ev.SrcNames = append(ev.SrcNames, C.GoString(info.src_names[k]))
				}
			}

			events = append(events, ev)
		}
	}

	cursor := passStartTime.UnixNano()
	for i := range events {
		events[i].TStart = cursor
		events[i].TEnd = cursor + events[i].TEnd
		cursor = events[i].TEnd
	}

	return events
}
```

- [ ] **Step 3: Commit**

```bash
git add llama/timing_bridge.go
git rm llama/profiler_bridge.go
git commit -m "llama: add timing CGO bridge for LlamaRunner"
```

---

## Task 8: Runner Integration — OllamaRunner

**Files:**
- Modify: `runner/ollamarunner/runner.go:332-391,642,1272-1277`
- Modify: `ml/backend/ggml/ggml.go:122,464`

- [ ] **Step 1: Remove old profiler code from ggml.go**

In `ml/backend/ggml/ggml.go`:

Remove the `profilerHandle cgo.Handle` field (line 122) from the `Backend` struct.

Remove the profilerHandle cleanup in `Close()` (line 464). Search for `profilerHandle` in Close and delete those lines.

Add a `graph` field to expose the last-computed graph for timing collection. In the `Context` struct (find it near `ComputeWithNotify`), ensure the `ggml_cgraph*` is accessible. The simplest approach: store `b.lastGraph = c.graph` on the Backend after compute. Check if there's already a suitable field.

Actually, the graph is on the `Context`, not the `Backend`. The timing bridge needs the graph. The cleanest way: have `CollectTiming` accept the graph from `ComputeWithNotify`'s context. Modify the `CollectTiming` signature to not need a stored graph — instead, the runner passes it.

Alternative: Store the last graph pointer on Backend:

```go
type Backend struct {
    // ... existing fields ...
    lastGraph *C.struct_ggml_cgraph  // set by ComputeWithNotify for timing collection
}
```

In `ComputeWithNotify`, after `graph_compute_async`, store:
```go
c.b.lastGraph = c.graph
```

Then `graphNodeAt` in `timing_bridge.go` uses `b.lastGraph`.

- [ ] **Step 2: Remove old profiler code from ollamarunner**

In `runner/ollamarunner/runner.go`:

Remove the `prof profiler.TraceCollector` field from Server struct (line 391).

Remove the profiler import: `"github.com/ollama/ollama/llm/profiler"` (will re-add with new API).

Remove all `s.prof.RecordPassStart(...)` calls (line 722).

Remove all `s.prof.RecordPassEnd(...)` calls (line 729).

Remove the `evalCallbackSetter` interface and the dynamic type assertion block (lines 1273-1277):
```go
// DELETE:
type evalCallbackSetter interface {
    SetEvalCallback(profiler.TraceCollector)
}
if setter, ok := s.model.Backend().(evalCallbackSetter); ok {
    setter.SetEvalCallback(s.prof)
}
```

Remove `s.prof = profiler.New(envconfig.TraceDir())` (line 1272).

Remove `s.prof.Flush(...)` calls and `s.prof.Close()` (line 1251).

- [ ] **Step 3: Add new tracing integration**

In `runner/ollamarunner/runner.go`:

Add import:
```go
"github.com/ollama/ollama/llm/profiler"
```

Add field to Server struct:
```go
traceWriter profiler.TraceWriter
```

In initialization (where old `s.prof` was set, around line 1272):
```go
s.traceWriter = profiler.NewWriter(envconfig.TraceDir())
if _, ok := s.traceWriter.(*profiler.JSONLWriter); ok {
    if tb, ok := s.model.Backend().(interface{ EnableTiming(bool) }); ok {
        tb.EnableTiming(true)
    }
}
```

In `computeBatch` (line 642 area), modify the batch processing:
```go
passStart := time.Now()
s.traceWriter.WritePassStart(activeBatch.id, len(batchInputs))
activeBatch.ctx.ComputeWithNotify(cb, activeBatch.modelOutput)
outputs := activeBatch.modelOutput.Floats()   // sync happens here
s.traceWriter.WritePassEnd(activeBatch.id)

// Pull timing after sync
if ct, ok := s.model.Backend().(interface {
    CollectTiming(time.Time) []profiler.OpEvent
}); ok {
    if events := ct.CollectTiming(passStart); len(events) > 0 {
        s.traceWriter.WriteOps(events)
    }
}
```

At request completion (where old `s.prof.Flush` was), add:
```go
s.traceWriter.Flush(requestID, s.modelPath)
```

At shutdown (where old `s.prof.Close()` was):
```go
s.traceWriter.Close()
```

- [ ] **Step 4: Commit**

```bash
git add runner/ollamarunner/runner.go ml/backend/ggml/ggml.go
git commit -m "ollamarunner: integrate native timing, remove eval callback"
```

---

## Task 9: Runner Integration — LlamaRunner

**Files:**
- Modify: `runner/llamarunner/runner.go:257-311,412,867`
- Modify: `llama/llama.go:164`

- [ ] **Step 1: Remove old profiler code from llama.go**

In `llama/llama.go`, remove the `profilerHandle cgo.Handle` field (line 164) and any associated cleanup.

- [ ] **Step 2: Remove old profiler code from llamarunner**

In `runner/llamarunner/runner.go`:

Remove `prof profiler.TraceCollector` field from Server struct (line 311).

Remove `batchID int` field if it was only used for profiler.

Remove profiler import (will re-add).

Remove `s.prof.RecordPassStart(...)` (line 503).

Remove `s.prof.RecordPassEnd(...)` (line 511).

Remove `s.lc.SetEvalCallback(s.prof)` (line 867).

Remove `s.prof = profiler.New(envconfig.TraceDir())` initialization.

Remove `s.prof.Flush(...)` calls (lines 732, 758).

- [ ] **Step 3: Add new tracing integration**

In `runner/llamarunner/runner.go`:

Add import:
```go
"github.com/ollama/ollama/llm/profiler"
```

Add field to Server struct:
```go
traceWriter profiler.TraceWriter
```

In initialization:
```go
s.traceWriter = profiler.NewWriter(envconfig.TraceDir())
if _, ok := s.traceWriter.(*profiler.JSONLWriter); ok {
    s.lc.EnableTiming(true)
}
```

In `processBatch` (line 412 area):
```go
passStart := time.Now()
s.traceWriter.WritePassStart(batchID, batch.NumTokens())
if err := s.lc.Decode(batch); err != nil {
    return err
}
s.lc.Synchronize()   // timing data readable after this
s.traceWriter.WritePassEnd(batchID)

// Pull timing
if events := s.lc.CollectTiming(passStart); len(events) > 0 {
    s.traceWriter.WriteOps(events)
}
```

At request completion:
```go
s.traceWriter.Flush(requestID, s.modelPath)
```

- [ ] **Step 4: Commit**

```bash
git add runner/llamarunner/runner.go llama/llama.go
git commit -m "llamarunner: integrate native timing, remove eval callback"
```

---

## Task 10: Final Cleanup

**Files:**
- Modify: `llama/patches/0018-ggml-Add-batch-size-hint.patch`
- Modify: `llama/llama.cpp/common/common.h:344` (remove cb_eval from common params)
- Modify: `llama/llama.cpp/src/llama-cparams.h:40` (remove cb_eval param field)

- [ ] **Step 1: Verify build compiles**

```bash
go build ./...
```

Fix any remaining compilation errors. Common issues:
- Missing imports (time package for runners)
- Leftover references to old profiler types (TraceCollector, RecordTensorStart, etc.)
- CGO compilation errors from C changes

- [ ] **Step 2: Regenerate patch file**

The patch at `llama/patches/0018-ggml-Add-batch-size-hint.patch` references `sched->callback_eval` at line 103. After the C layer changes, this patch context will be stale.

Regenerate the patch:
```bash
# The patch modifies ggml-backend.cpp to add batch_size support.
# After our changes to compute_splits, the context around the patched
# area has changed. Regenerate from the current state.
```

Check if the patch still applies. If not, manually update the context lines in the patch file to match the new code (the `callback_eval` reference should now be gone).

- [ ] **Step 3: Clean up llama.cpp common params**

In `llama/llama.cpp/common/common.h` (line 344), remove `cb_eval` and `cb_eval_user_data` fields if they exist in the common params struct.

In `llama/llama.cpp/src/llama-cparams.h` (line 40), remove the `cb_eval` param field.

Search for any remaining references:
```bash
grep -r "cb_eval\|callback_eval\|set_eval_callback" llama/llama.cpp/
```

Remove all remaining references.

- [ ] **Step 4: Run profiler package tests**

```bash
go test ./llm/profiler/ -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "cleanup: remove all eval callback remnants, regenerate patch"
```

---

## Task 11: Build Verification and Integration Test

- [ ] **Step 1: Full build**

```bash
go build ./...
```

Expected: Clean build with no errors.

- [ ] **Step 2: Run all Go tests**

```bash
go test ./...
```

Expected: All existing tests pass. New profiler tests pass.

- [ ] **Step 3: Manual integration test — no tracing**

```bash
# Run server without OLLAMA_TRACE_DIR
go run . serve
# In another terminal, run a simple inference
curl http://localhost:11434/api/generate -d '{"model":"qwen3:0.6b","prompt":"hello","stream":false}'
```

Expected: Normal inference, no trace files, no crashes, no performance regression.

- [ ] **Step 4: Manual integration test — with tracing**

```bash
mkdir -p /tmp/traces
OLLAMA_TRACE_DIR=/tmp/traces go run . serve
# Run inference
curl http://localhost:11434/api/generate -d '{"model":"qwen3:0.6b","prompt":"hello","stream":false}'
# Check trace output
ls /tmp/traces/
cat /tmp/traces/trace_*.jsonl | head -20
```

Expected:
- JSONL file produced in `/tmp/traces/`
- File contains `pass_start`, multiple `op` lines, `pass_end`
- Each `op` line has: type, pass, seq, op, name, srcs, shape, dtype, backend, t_start, t_end
- t_start < t_end for each op
- Sequential seq IDs within each pass

- [ ] **Step 5: Verify trace-analyzer compatibility**

```bash
# Feed new JSONL to existing trace-analyzer
npx --prefix tools/trace-analyzer serve /tmp/traces/trace_*.jsonl
# Open browser at http://localhost:3000
```

Expected: DAG, Timeline, Heatmap, Replay views all render correctly.

- [ ] **Step 6: Performance comparison**

```bash
# Baseline (no tracing)
ollama-bench --model qwen3:0.6b --n-prompt 128 --n-gen 64

# With tracing
OLLAMA_TRACE_DIR=/tmp/traces ollama-bench --model qwen3:0.6b --n-prompt 128 --n-gen 64
```

Expected: <5% degradation with tracing enabled.

- [ ] **Step 7: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "test: verify native timing integration"
```
