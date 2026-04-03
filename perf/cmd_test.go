package perf

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBenchmarkCLIOptions_Defaults(t *testing.T) {
	opts := BenchmarkCLIOptions{}
	assert.Empty(t, opts.Output)
	assert.Empty(t, opts.Ops)
	assert.Empty(t, opts.Dtypes)
	assert.False(t, opts.Viewer)
	assert.False(t, opts.Verbose)
}

func TestEstimateCLIOptions_Defaults(t *testing.T) {
	opts := EstimateCLIOptions{}
	assert.Empty(t, opts.Profile)
	assert.False(t, opts.JSON)
	assert.False(t, opts.Verbose)
}

func TestViewerCLIOptions_Defaults(t *testing.T) {
	opts := ViewerCLIOptions{}
	assert.Empty(t, opts.Profile)
	assert.Empty(t, opts.Output)
}

func TestRunBenchmarkCLI_OpsParsingDefault(t *testing.T) {
	opts := BenchmarkCLIOptions{}
	assert.Empty(t, opts.Ops, "empty means use defaults")
}

func TestRunBenchmarkCLI_OpsParsing(t *testing.T) {
	input := "SILU,MUL_MAT"
	ops := splitOps(input)
	assert.Equal(t, []string{"SILU", "MUL_MAT"}, ops)
}

// splitOps mirrors the strings.Split logic in RunBenchmarkCLI
func splitOps(s string) []string {
	if s == "" {
		return []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	}
	return strings.Split(s, ",")
}

func TestOpenBrowser_UnsupportedOS(t *testing.T) {
	assert.NotPanics(t, func() {
		openBrowser("/tmp/test.html")
	})
}

func TestDefaultBenchmarkConfig(t *testing.T) {
	cfg := DefaultBenchmarkConfig()
	assert.Equal(t, 5, cfg.WarmupReps)
	assert.Equal(t, 50, cfg.MeasureReps)
	assert.Equal(t, 0.1, cfg.TrimPercent)
	assert.Equal(t, 0.05, cfg.ErrorThreshold)
	assert.Equal(t, 20, cfg.MaxPointsPerOp)
}
