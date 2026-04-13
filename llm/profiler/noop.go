package profiler

// NoopWriter implements TraceWriter with zero overhead.
type NoopWriter struct{}

func (n *NoopWriter) WriteOps(_ []OpEvent)           {}
func (n *NoopWriter) WritePassStart(_ int, _ int)    {}
func (n *NoopWriter) WritePassEnd(_ int)             {}
func (n *NoopWriter) Flush(_ string, _ string) error { return nil }
func (n *NoopWriter) Close() error                   { return nil }
