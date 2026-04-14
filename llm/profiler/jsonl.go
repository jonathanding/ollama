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
	mu           sync.Mutex
	wg           sync.WaitGroup
	lines        [][]byte
	outDir       string
	seqID        int
	inputTokens  int
	outputTokens int
}

func newJSONLWriter(outDir string) *JSONLWriter {
	return &JSONLWriter{outDir: outDir}
}

func (w *JSONLWriter) WriteOps(ops []OpEvent) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, op := range ops {
		op.SeqID = w.seqID
		w.seqID++
		line, _ := json.Marshal(op)
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
	if nTokens > 1 {
		w.inputTokens += nTokens
	} else {
		w.outputTokens += nTokens
	}
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
	inputTokens := w.inputTokens
	outputTokens := w.outputTokens
	w.inputTokens = 0
	w.outputTokens = 0
	w.mu.Unlock()

	if len(lines) == 0 {
		return nil
	}

	meta, _ := json.Marshal(map[string]any{
		"type":          "meta",
		"model":         model,
		"request_id":    requestID,
		"ts":            time.Now().UnixMilli(),
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
	})

	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
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
		f.Write(meta)
		f.Write([]byte{'\n'})
		for _, line := range lines {
			f.Write(line)
			f.Write([]byte{'\n'})
		}
	}()
	return nil
}

func (w *JSONLWriter) Close() error {
	w.wg.Wait()
	return nil
}
