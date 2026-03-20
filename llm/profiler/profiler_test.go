package profiler_test

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

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
