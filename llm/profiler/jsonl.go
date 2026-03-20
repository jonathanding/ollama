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
