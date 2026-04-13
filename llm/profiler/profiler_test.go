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
