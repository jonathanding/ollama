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
