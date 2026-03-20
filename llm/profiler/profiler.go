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
