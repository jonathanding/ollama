# Ollama Execution Profiling & Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in per-operator execution tracing to LlamaRunner that writes JSONL trace files suitable for DAG visualization, with zero overhead when disabled.

**Architecture:** Expose GGML's existing `ggml_backend_sched_eval_callback` via a minimal C++ patch (~10 lines) and a CGO bridge in `llama/profiler_bridge.go`. The `llm/profiler` package is pure Go (no CGO) and provides a `TraceCollector` interface. LlamaRunner injects a `JSONLTraceBuffer` when `OLLAMA_TRACE_DIR` is set, otherwise a zero-cost `NoopCollector`. Each request produces one `.jsonl` file, flushed asynchronously after inference completes.

**Tech Stack:** Go 1.21+, CGO, C++17 (vendored llama.cpp), `runtime/cgo` for safe Go↔C pointer passing, `encoding/json`, `sync.Mutex`.

**Spec:** `docs/superpowers/specs/2026-03-19-ollama-profiling-tracing-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `llama/llama.cpp/include/llama.h` | Modify | Declare `llama_context_set_eval_callback()` |
| `llama/llama.cpp/src/llama.cpp` | Modify | Implement it via `ctx->get_sched()` |
| `llama/profiler_bridge.go` | **Create** | `//export GoProfilerCallback` + C bridge + `extractTensorInfo`. **Must be a separate file** — CGO forbids `//export` and C definitions in the same file. |
| `llama/llama.go` | Modify | Add `profilerHandle cgo.Handle` to `Context` struct; add `SetEvalCallback()` method |
| `llm/profiler/profiler.go` | **Create** | `TraceCollector` interface (pure Go, no CGO), `OpEvent`, `TensorInfo`, `New()` factory |
| `llm/profiler/noop.go` | **Create** | `NoopCollector` — all methods are no-ops |
| `llm/profiler/jsonl.go` | **Create** | `JSONLTraceBuffer` — collects events in memory, flushes to disk async |
| `llm/profiler/profiler_test.go` | **Create** | Unit tests for buffer, serialization, flush |
| `envconfig/config.go` | Modify | Add `TraceDir = String("OLLAMA_TRACE_DIR")` |
| `runner/llamarunner/runner.go` | Modify | Add `profiler` to `Server`; hook `loadModel`, `processBatch`, `completion` |

---

## Task 1: C++ patch — expose `llama_context_set_eval_callback`

**Files:**
- Modify: `llama/llama.cpp/include/llama.h` (after line 943, after `llama_set_abort_callback`)
- Modify: `llama/llama.cpp/src/llama.cpp` (find `void llama_set_abort_callback(` and add after it)

**Background:**
- `ggml_backend_sched_set_eval_callback` exists at `ml/backend/ggml/ggml/include/ggml-backend.h:354`
- `llama_context` exposes a public accessor `get_sched()` at `llama/llama.cpp/src/llama-context.h:48` — use this, NOT `ctx->sched.get()` (private member)
- The existing `llama_set_abort_callback` at line 943 of `llama.h` is the style reference

- [ ] **Step 1.1: Add declaration to `llama.h`**

  Open `llama/llama.cpp/include/llama.h`. Find line 943:
  ```c
  LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);
  ```
  Insert immediately after it:
  ```c
  LLAMA_API void llama_context_set_eval_callback(struct llama_context * ctx, ggml_backend_sched_eval_callback callback, void * user_data);
  ```

- [ ] **Step 1.2: Add implementation to `llama.cpp`**

  Open `llama/llama.cpp/src/llama.cpp`. Search for the exact line `void llama_set_abort_callback(` to locate the implementation block. Add the new function immediately after it:
  ```cpp
  void llama_context_set_eval_callback(struct llama_context * ctx, ggml_backend_sched_eval_callback callback, void * user_data) {
      ggml_backend_sched_set_eval_callback(ctx->get_sched(), callback, user_data);
  }
  ```
  `ctx->get_sched()` is declared in `llama-context.h:48` and returns `ggml_backend_sched_t`.

- [ ] **Step 1.3: Verify it compiles**

  ```bash
  cd /c/workspace/myollama
  go build ./llama/
  ```

  Expected: no errors. Troubleshooting:
  - `ctx->get_sched()` not found → check `#include "llama-context.h"` is present in the .cpp file
  - `ggml_backend_sched_set_eval_callback` not found → check `ggml-backend.h` is included

- [ ] **Step 1.4: Commit**

  ```bash
  git add llama/llama.cpp/include/llama.h llama/llama.cpp/src/llama.cpp
  git commit -m "llama: expose ggml eval callback via llama_context_set_eval_callback"
  ```

---

## Task 2: Go profiler package — interface, noop, and tests

**Files:**
- Create: `llm/profiler/profiler.go`
- Create: `llm/profiler/noop.go`
- Create: `llm/profiler/profiler_test.go`

**Interface design:** The interface uses `RecordTensorStart(uintptr, int64)` / `RecordTensorEnd(uintptr, TensorInfo, int64)` instead of a single `HandleTensor(unsafe.Pointer, bool)`. This keeps `llm/profiler` as pure Go with no CGO dependency. All C struct access lives in `llama/profiler_bridge.go`, which calls these methods with plain Go types. `JSONLTraceBuffer` comes in Task 3; this task only covers the interface and `NoopCollector`.

- [ ] **Step 2.1: Write failing tests**

  Create `llm/profiler/profiler_test.go`:

  ```go
  package profiler_test

  import (
      "testing"

      "github.com/ollama/ollama/llm/profiler"
  )

  // TestNoopCollectorImplementsInterface verifies NoopCollector satisfies TraceCollector at compile time.
  func TestNoopCollectorImplementsInterface(t *testing.T) {
      var _ profiler.TraceCollector = &profiler.NoopCollector{}
  }

  // TestNewReturnsNoopWhenDirEmpty verifies New("") returns a working NoopCollector.
  func TestNewReturnsNoopWhenDirEmpty(t *testing.T) {
      c := profiler.New("")
      if c == nil {
          t.Fatal("New should never return nil")
      }
      // None of these should panic
      c.RecordPassStart(0, 0)
      c.RecordTensorStart(0, 0)
      c.RecordTensorEnd(0, profiler.TensorInfo{}, 0)
      c.RecordPassEnd(0, 0)
      if err := c.Flush("req-1", "model"); err != nil {
          t.Errorf("Flush on NoopCollector returned non-nil: %v", err)
      }
      if err := c.Close(); err != nil {
          t.Errorf("Close on NoopCollector returned non-nil: %v", err)
      }
  }
  ```

- [ ] **Step 2.2: Run tests — expect compile failure**

  ```bash
  go test ./llm/profiler/...
  ```
  Expected: `cannot find package "github.com/ollama/ollama/llm/profiler"`

- [ ] **Step 2.3: Create `llm/profiler/profiler.go`**

  ```go
  package profiler

  // TraceCollector is the unified interface for all instrumentation paths.
  // Phase 1: LlamaRunner feeds events via the CGO bridge in llama/profiler_bridge.go.
  // Phase 2: OllamaRunner will feed events from Go-layer operator calls.
  //
  // This package has NO CGO imports. All C struct access is in llama/profiler_bridge.go,
  // which extracts tensor metadata into TensorInfo and calls RecordTensorStart/End.
  type TraceCollector interface {
      // RecordTensorStart is called when a GGML node is about to be dispatched.
      // ptr is the C tensor pointer cast to uintptr — used as a map key only,
      // never dereferenced here.
      RecordTensorStart(ptr uintptr, tStart int64)

      // RecordTensorEnd is called after dispatch. info holds tensor metadata
      // extracted by the CGO bridge.
      RecordTensorEnd(ptr uintptr, info TensorInfo, tEnd int64)

      RecordPassStart(passID int, nTokens int)
      RecordPassEnd(passID int, nNodes int)

      // Flush writes all buffered events to trace_<requestID>_<ts>.jsonl.
      // Happens in a background goroutine; buffer is cleared synchronously.
      // Always returns nil; write errors are logged via slog.
      Flush(requestID string, model string) error

      Close() error
  }

  // TensorInfo holds tensor metadata extracted by the CGO bridge from a ggml_tensor*.
  type TensorInfo struct {
      Op       string
      Name     string
      SrcNames []string
      OutShape []int64
      DType    string
      Backend  string
  }

  // OpEvent represents one GGML operator node execution.
  // JSON field names define the JSONL output format.
  type OpEvent struct {
      Type     string   `json:"type"`    // always "op"
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

  // New returns a JSONLTraceBuffer if outDir is non-empty, otherwise NoopCollector.
  func New(outDir string) TraceCollector {
      if outDir == "" {
          return &NoopCollector{}
      }
      return newJSONLTraceBuffer(outDir)
  }
  ```

- [ ] **Step 2.4: Create `llm/profiler/noop.go`**

  ```go
  package profiler

  // NoopCollector implements TraceCollector with zero overhead.
  // Used when OLLAMA_TRACE_DIR is not set.
  type NoopCollector struct{}

  func (n *NoopCollector) RecordTensorStart(_ uintptr, _ int64)             {}
  func (n *NoopCollector) RecordTensorEnd(_ uintptr, _ TensorInfo, _ int64) {}
  func (n *NoopCollector) RecordPassStart(_ int, _ int)                      {}
  func (n *NoopCollector) RecordPassEnd(_ int, _ int)                        {}
  func (n *NoopCollector) Flush(_ string, _ string) error                    { return nil }
  func (n *NoopCollector) Close() error                                       { return nil }
  ```

- [ ] **Step 2.5: Run tests — should pass**

  ```bash
  go test ./llm/profiler/...
  ```
  Expected: `PASS`

- [ ] **Step 2.6: Commit**

  ```bash
  git add llm/profiler/profiler.go llm/profiler/noop.go llm/profiler/profiler_test.go
  git commit -m "llm/profiler: add TraceCollector interface and NoopCollector"
  ```

---

## Task 3: JSONLTraceBuffer — in-memory collection and async flush

**Files:**
- Create: `llm/profiler/jsonl.go`
- Modify: `llm/profiler/profiler_test.go` (add buffer/flush tests)

**Background:**
- `JSONLTraceBuffer` is per-`llama.Context`, not a global. No cross-request sharing.
- `RecordTensorStart/End` are called from the C thread during `lc.Decode()`. They take a mutex.
- The pending map key is `uintptr(tensorPtr)` — tensor arena pointers are stable for the duration of one `Decode()` call.

- [ ] **Step 3.1: Add buffer and serialization tests**

  Append to `llm/profiler/profiler_test.go`:

  ```go
  import (
      "encoding/json"
      "os"
      "strings"
      "time"
  )

  // TestJSONLFlushWritesFile verifies Flush creates a .jsonl file in the output dir.
  func TestJSONLFlushWritesFile(t *testing.T) {
      dir := t.TempDir()
      buf := profiler.New(dir)

      buf.RecordPassStart(0, 512)
      buf.RecordPassEnd(0, 100)
      if err := buf.Flush("req-abc", "llama3:8b"); err != nil {
          t.Fatalf("Flush returned error: %v", err)
      }

      time.Sleep(50 * time.Millisecond) // let async goroutine finish

      entries, err := os.ReadDir(dir)
      if err != nil {
          t.Fatal(err)
      }
      if len(entries) != 1 {
          t.Fatalf("expected 1 file, got %d", len(entries))
      }
      if !strings.HasSuffix(entries[0].Name(), ".jsonl") {
          t.Errorf("expected .jsonl file, got %s", entries[0].Name())
      }
  }

  // TestJSONLFlushClearsBuffer verifies buffer is empty after Flush.
  func TestJSONLFlushClearsBuffer(t *testing.T) {
      dir := t.TempDir()
      buf := profiler.New(dir)
      buf.RecordPassStart(0, 10)
      buf.Flush("req-1", "model")
      time.Sleep(50 * time.Millisecond)

      buf.Flush("req-2", "model") // second flush with empty buffer
      time.Sleep(50 * time.Millisecond)

      entries, _ := os.ReadDir(dir)
      if len(entries) < 2 {
          t.Errorf("expected 2 trace files, got %d", len(entries))
      }
  }

  // TestOpEventJSONFields verifies JSON keys match the JSONL spec.
  func TestOpEventJSONFields(t *testing.T) {
      ev := profiler.OpEvent{
          Type: "op", PassID: 1, SeqID: 2,
          Op: "MUL_MAT", Name: "blk.3.attn_q",
          SrcNames: []string{"blk.3.attn_norm"},
          OutShape: []int64{512, 4096},
          DType: "f16", Backend: "CUDA0",
          TStart: 1000, TEnd: 2000,
      }
      data, _ := json.Marshal(ev)
      s := string(data)
      for _, want := range []string{`"type"`, `"pass"`, `"seq"`, `"op"`, `"name"`,
          `"srcs"`, `"shape"`, `"dtype"`, `"backend"`, `"t_start"`, `"t_end"`} {
          if !strings.Contains(s, want) {
              t.Errorf("JSON missing field %s in: %s", want, s)
          }
      }
  }
  ```

- [ ] **Step 3.2: Run tests — expect compile failure on `newJSONLTraceBuffer`**

  ```bash
  go test ./llm/profiler/...
  ```
  Expected: `undefined: newJSONLTraceBuffer`

- [ ] **Step 3.3: Create `llm/profiler/jsonl.go`**

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

  type JSONLTraceBuffer struct {
      mu      sync.Mutex
      lines   [][]byte          // serialized JSONL lines for the current request
      pending map[uintptr]int64 // tensorPtr uintptr → t_start (ns)
      outDir  string
      passID  int
      seqID   int
  }

  func newJSONLTraceBuffer(outDir string) *JSONLTraceBuffer {
      return &JSONLTraceBuffer{
          outDir:  outDir,
          pending: make(map[uintptr]int64),
      }
  }

  func (b *JSONLTraceBuffer) RecordTensorStart(ptr uintptr, tStart int64) {
      b.mu.Lock()
      b.pending[ptr] = tStart
      b.mu.Unlock()
  }

  func (b *JSONLTraceBuffer) RecordTensorEnd(ptr uintptr, info TensorInfo, tEnd int64) {
      b.mu.Lock()
      tStart, ok := b.pending[ptr]
      if ok {
          delete(b.pending, ptr)
      }
      seqID := b.seqID
      b.seqID++
      passID := b.passID
      b.mu.Unlock()

      if !ok {
          return
      }

      ev := OpEvent{
          Type: "op", PassID: passID, SeqID: seqID,
          Op: info.Op, Name: info.Name,
          SrcNames: info.SrcNames, OutShape: info.OutShape,
          DType: info.DType, Backend: info.Backend,
          TStart: tStart, TEnd: tEnd,
      }
      line, _ := json.Marshal(ev)
      b.mu.Lock()
      b.lines = append(b.lines, line)
      b.mu.Unlock()
  }

  func (b *JSONLTraceBuffer) RecordPassStart(passID int, nTokens int) {
      b.mu.Lock()
      b.passID = passID
      b.seqID = 0
      b.mu.Unlock()
      line, _ := json.Marshal(map[string]any{
          "type": "pass_start", "pass": passID, "n_tokens": nTokens,
          "ts": time.Now().UnixMilli(),
      })
      b.mu.Lock()
      b.lines = append(b.lines, line)
      b.mu.Unlock()
  }

  func (b *JSONLTraceBuffer) RecordPassEnd(passID int, nNodes int) {
      line, _ := json.Marshal(map[string]any{
          "type": "pass_end", "pass": passID, "n_nodes": nNodes,
          "ts": time.Now().UnixMilli(),
      })
      b.mu.Lock()
      b.lines = append(b.lines, line)
      b.mu.Unlock()
  }

  // Flush hands buffered lines to a background goroutine and clears the buffer.
  // Always returns nil; write errors are logged.
  func (b *JSONLTraceBuffer) Flush(requestID, model string) error {
      b.mu.Lock()
      lines := b.lines
      b.lines = nil
      b.mu.Unlock()

      go func() {
          ts := time.Now().UnixMilli()
          safe := strings.Map(func(r rune) rune {
              if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
                  (r >= '0' && r <= '9') || r == '-' || r == '_' {
                  return r
              }
              return '_'
          }, requestID)
          fname := filepath.Join(b.outDir, fmt.Sprintf("trace_%s_%d.jsonl", safe, ts))
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

  func (b *JSONLTraceBuffer) Close() error { return nil }
  ```

- [ ] **Step 3.4: Run tests — should pass**

  ```bash
  go test ./llm/profiler/...
  ```
  Expected: `PASS`

- [ ] **Step 3.5: Commit**

  ```bash
  git add llm/profiler/
  git commit -m "llm/profiler: add JSONLTraceBuffer with async flush"
  ```

---

## Task 4: envconfig — add `OLLAMA_TRACE_DIR`

**Files:**
- Modify: `envconfig/config.go` (near line 223, where `LLMLibrary` and `Editor` are defined)

- [ ] **Step 4.1: Add the variable**

  Find this block in `envconfig/config.go`:
  ```go
  LLMLibrary = String("OLLAMA_LLM_LIBRARY")
  Editor     = String("OLLAMA_EDITOR")
  ```
  Add after it:
  ```go
  // TraceDir returns the directory for writing inference trace files.
  // When empty (the default), profiling is disabled and zero overhead is incurred.
  // Set OLLAMA_TRACE_DIR=/path/to/dir to enable per-request JSONL trace output.
  TraceDir = String("OLLAMA_TRACE_DIR")
  ```

- [ ] **Step 4.2: Verify build**

  ```bash
  go build ./envconfig/...
  ```
  Expected: no errors.

- [ ] **Step 4.3: Commit**

  ```bash
  git add envconfig/config.go
  git commit -m "envconfig: add OLLAMA_TRACE_DIR for inference profiling"
  ```

---

## Task 5: CGO bridge — wire Go profiler to GGML eval callback

**Files:**
- Create: `llama/profiler_bridge.go` ← **separate file required by CGO**
- Modify: `llama/llama.go` (add field + method to `Context`)

**Why a separate file?** CGO rule: a `.go` file with `//export` annotations cannot have C definitions in its `import "C"` preamble. `llama/llama.go` already has a large preamble with definitions. Therefore `GoProfilerCallback` and `extractTensorInfo` must live in `llama/profiler_bridge.go`. Both files are in `package llama`, so they share the `Context` type.

- [ ] **Step 5.1: Create `llama/profiler_bridge.go`**

  ```go
  package llama

  /*
  #include <stdlib.h>
  #include "ggml.h"
  #include "ggml-backend.h"
  #include "llama.h"

  extern void GoProfilerCallback(GoUintptr handle, void* tensorPtr, _Bool ask);

  static _Bool profilerEvalCallbackBridge(struct ggml_tensor* t, _Bool ask, void* user_data) {
      GoProfilerCallback((GoUintptr)(uintptr_t)user_data, (void*)t, ask);
      return ask;
  }
  */
  import "C"

  import (
      "runtime/cgo"
      "time"
      "unsafe"

      "github.com/ollama/ollama/llm/profiler"
  )

  // GoProfilerCallback is called from the C eval callback bridge on the C thread.
  // ask=true: tensor about to be dispatched. ask=false: tensor dispatched.
  //
  //export GoProfilerCallback
  func GoProfilerCallback(handle C.GoUintptr, tensorPtr unsafe.Pointer, ask C.bool) {
      h := cgo.Handle(handle)
      col, ok := h.Value().(profiler.TraceCollector)
      if !ok {
          return
      }
      ptr := uintptr(tensorPtr)
      now := time.Now().UnixNano()
      if bool(ask) {
          col.RecordTensorStart(ptr, now)
          return
      }
      t := (*C.struct_ggml_tensor)(tensorPtr)
      col.RecordTensorEnd(ptr, extractTensorInfo(t), now)
  }

  // extractTensorInfo reads metadata from a C ggml_tensor pointer.
  // Safe to call only during Decode() — tensor arena pointers are stable
  // for the duration of a single llama_decode() call.
  // This function must stay in this file (CGO file) since it accesses C structs.
  func extractTensorInfo(t *C.struct_ggml_tensor) profiler.TensorInfo {
      info := profiler.TensorInfo{
          Op:   C.GoString(C.ggml_op_name(t.op)),
          Name: C.GoString(&t.name[0]),
      }

      // Output shape: ne[0..3]; trim trailing 1s
      shape := make([]int64, 4)
      for i := 0; i < 4; i++ {
          shape[i] = int64(t.ne[i])
      }
      for len(shape) > 1 && shape[len(shape)-1] == 1 {
          shape = shape[:len(shape)-1]
      }
      info.OutShape = shape

      // Data type name. CGO mangles C field "type" to "_type".
      info.DType = C.GoString(C.ggml_type_name(t._type))

      // Backend (device) where this tensor lives
      if t.buffer != nil {
          info.Backend = C.GoString(C.ggml_backend_buffer_name(t.buffer))
      }

      // Source tensor names → DAG edges.
      // GGML_MAX_SRC = 10; stop at first NULL.
      for i := 0; i < 10; i++ {
          src := t.src[i]
          if src == nil {
              break
          }
          info.SrcNames = append(info.SrcNames, C.GoString(&src.name[0]))
      }

      return info
  }
  ```

  **Compile-time notes:**
  - `t._type`: CGO renames the C field `type` to `_type`. If this fails, use `C.ggml_get_type(t)` if available.
  - `C.GGML_MAX_SRC` may not be available as a CGO constant; the hardcoded `10` is the actual value from `ggml.h`.
  - `ggml_backend_buffer_name`: declared in `ggml-backend.h`. If the vendored version lacks it, leave `Backend` as `""` for now.

- [ ] **Step 5.2: Add `profilerHandle` field and `SetEvalCallback` to `llama/llama.go`**

  Find the `Context` struct at line 161:
  ```go
  type Context struct {
  	c          *C.struct_llama_context
  	numThreads int
  }
  ```
  Add the new field:
  ```go
  type Context struct {
  	c               *C.struct_llama_context
  	numThreads      int
  	profilerHandle  cgo.Handle // non-zero when profiling is active
  }
  ```

  Then add `SetEvalCallback` as a new method (add it after the existing `Synchronize` method):
  ```go
  // SetEvalCallback registers a per-operator callback for inference profiling.
  // Pass a non-nil TraceCollector to enable; nil to disable (registers NULL callback, zero overhead).
  // Call after NewContextWithModel and before the first Decode.
  func (c *Context) SetEvalCallback(col profiler.TraceCollector) {
  	if c.profilerHandle != 0 {
  		c.profilerHandle.Delete()
  		c.profilerHandle = 0
  	}
  	if col == nil {
  		C.llama_context_set_eval_callback(c.c, nil, nil)
  		return
  	}
  	h := cgo.NewHandle(col)
  	c.profilerHandle = h
  	C.llama_context_set_eval_callback(c.c,
  		C.profilerEvalCallbackBridge,
  		C.GoUintptr(uintptr(h)))
  }
  ```

  Add to the imports in `llama/llama.go`:
  ```go
  "github.com/ollama/ollama/llm/profiler"
  ```

  Note: `profilerEvalCallbackBridge` is defined in `profiler_bridge.go` (same package), so it's accessible here without any import. The function pointer type matches because both files are compiled together as package `llama`.

- [ ] **Step 5.3: Verify compilation**

  ```bash
  go build ./llama/
  ```

- [ ] **Step 5.4: Run existing tests**

  ```bash
  go test ./llama/...
  ```
  Expected: `PASS` (existing grammar tests still pass).

- [ ] **Step 5.5: Commit**

  ```bash
  git add llama/profiler_bridge.go llama/llama.go
  git commit -m "llama: add SetEvalCallback CGO bridge for inference profiling"
  ```

---

## Task 6: LlamaRunner integration — hook profiler into inference loop

**Files:**
- Modify: `runner/llamarunner/runner.go`

**Key line references (verified):**
- Line 256: `type Server struct`
- Line 829: `func (s *Server) loadModel(...)`
- Line 847: `s.lc, err = llama.NewContextWithModel(...)` ← inject profiler here
- Line 405: `func (s *Server) processBatch(...)`
- Line 494: `if err := s.lc.Decode(batch)` ← wrap with pass tracking
- Line 622: `func (s *Server) completion(...)` ← emit request events + flush

**Thread safety:** `processBatch` is called under `s.mu` (line 288). This means `RecordTensorStart/End` will be called from a single goroutine at a time — the `JSONLTraceBuffer`'s internal mutex is for correctness when `Flush` runs concurrently from the HTTP handler goroutine.

- [ ] **Step 6.1: Add `profiler` and `batchID` fields to `Server` struct**

  Find the `Server` struct at line 256. Add at the end (before closing `}`):
  ```go
  // profiler collects per-operator trace events. NoopCollector when OLLAMA_TRACE_DIR is unset.
  profiler profiler.TraceCollector

  // batchID is incremented each processBatch call, used as PassID in traces.
  batchID int
  ```

  Add imports:
  ```go
  "github.com/ollama/ollama/envconfig"
  "github.com/ollama/ollama/llm/profiler"
  ```

- [ ] **Step 6.2: Initialize profiler in `loadModel`**

  In `loadModel` (line 829), find:
  ```go
  s.lc, err = llama.NewContextWithModel(s.model, ctxParams)
  if err != nil {
  	panic(err)
  }
  ```
  Add immediately after:
  ```go
  s.profiler = profiler.New(envconfig.TraceDir())
  s.lc.SetEvalCallback(s.profiler)
  ```

- [ ] **Step 6.3: Wrap Decode with pass tracking in `processBatch`**

  In `processBatch`, find the decode section (around line 493):
  ```go
  t := time.Now()
  if err := s.lc.Decode(batch); err != nil {
  	return fmt.Errorf("failed to decode batch: %w", err)
  }

  if numOutputs > 0 {
  	s.lc.Synchronize()
  }
  ```
  Replace with:
  ```go
  t := time.Now()
  batchID := s.batchID
  s.batchID++
  s.profiler.RecordPassStart(batchID, batch.NumTokens())
  if err := s.lc.Decode(batch); err != nil {
  	return fmt.Errorf("failed to decode batch: %w", err)
  }

  if numOutputs > 0 {
  	s.lc.Synchronize()
  }
  s.profiler.RecordPassEnd(batchID, 0)
  ```

- [ ] **Step 6.4: Add request ID and Flush in `completion` handler**

  At the top of the `completion` handler (line 622), after decoding the request body:
  ```go
  requestID := fmt.Sprintf("%d", time.Now().UnixNano())
  ```

  Find the final-response block (around line 733) where `!ok` on the channel:
  ```go
  } else {
  	if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
  		Done: true,
  		// ...
  	}); err != nil {
  		http.Error(...)
  	}
  	return
  }
  ```
  Add `s.profiler.Flush(requestID, s.modelPath)` before the `return`:
  ```go
  } else {
  	if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
  		Done:               true,
  		DoneReason:         seq.doneReason,
  		PromptEvalCount:    seq.numPromptInputs,
  		PromptEvalDuration: seq.processingDuration,
  		EvalCount:          seq.numDecoded,
  		EvalDuration:       seq.generationDuration,
  	}); err != nil {
  		http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
  	}
  	s.profiler.Flush(requestID, s.modelPath)
  	return
  }
  ```

  Also flush on connection-closed early return (around line 717):
  ```go
  case <-r.Context().Done():
  	close(seq.quit)
  	s.profiler.Flush(requestID, s.modelPath)
  	return
  ```

- [ ] **Step 6.5: Build the runner**

  ```bash
  go build ./runner/...
  ```

- [ ] **Step 6.6: Run unit tests**

  ```bash
  go test ./runner/llamarunner/... ./llm/... ./llama/...
  ```
  Expected: all pass.

- [ ] **Step 6.7: Commit**

  ```bash
  git add runner/llamarunner/runner.go
  git commit -m "runner/llamarunner: integrate profiler for per-operator tracing"
  ```

---

## Task 7: End-to-end verification

Requires GPU and `llama3.2:1b` model (~1.7 GB, downloaded automatically on first run).

- [ ] **Step 7.1: Full unit test suite**

  ```bash
  go test ./...
  ```
  All must pass before proceeding.

- [ ] **Step 7.2: Regression test — profiling disabled (normal inference)**

  ```bash
  go test -tags=integration -timeout=5m -run TestBlueSky ./integration
  ```
  Expected: `PASS`. Confirms no inference regression when `OLLAMA_TRACE_DIR` is unset.

- [ ] **Step 7.3: Smoke test — profiling enabled**

  ```bash
  mkdir -p /tmp/traces
  OLLAMA_TRACE_DIR=/tmp/traces \
    go test -tags=integration -timeout=5m -run TestBlueSky ./integration
  ```
  Expected: `PASS` and at least one `.jsonl` file in `/tmp/traces/`.

- [ ] **Step 7.4: Inspect trace content**

  ```bash
  # Preview first 20 lines
  cat /tmp/traces/trace_*.jsonl | head -20

  # Operator coverage
  cat /tmp/traces/trace_*.jsonl | jq -r 'select(.type=="op")|.op' | sort | uniq -c | sort -rn
  # Expected top ops: MUL_MAT, RMS_NORM, CPY, GET_ROWS, ADD

  # DAG edge check — srcs should reference real tensor names
  cat /tmp/traces/trace_*.jsonl | jq 'select(.type=="op")|{name,srcs}' | head -20

  # Backend check — most ops should be on GPU
  cat /tmp/traces/trace_*.jsonl | jq -r 'select(.type=="op")|.backend' | sort | uniq -c
  ```

- [ ] **Step 7.5: Final commit**

  ```bash
  git add -p
  git commit -m "profiler: complete Phase 1 LlamaRunner instrumentation"
  ```

---

## Known Caveats for Implementer

1. **`t._type` in CGO:** C struct field `type` is accessible in Go as `_type`. If the compiler rejects it, use `C.ggml_get_type(t)` if available in the vendored ggml.h.

2. **`GGML_MAX_SRC` as CGO constant:** The macro may not be directly accessible via `C.GGML_MAX_SRC`. The plan uses the literal `10`, which is the value defined in `ggml.h` and unlikely to change.

3. **`ggml_backend_buffer_name`:** If missing in the vendored version, leave `Backend` as `""` initially. The tensor still records all other fields.

4. **`batchID` visibility in `processBatch`:** The field is on `Server` and accessed under `s.mu`, which `processBatch` already holds. No additional locking needed.

5. **`llm.CompletionRequest` has no ID field:** The plan generates `requestID` from `time.Now().UnixNano()` — sufficient for unique file naming within a session.
