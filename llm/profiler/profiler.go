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
