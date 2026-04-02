package perf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsGGUFPath(t *testing.T) {
	assert.True(t, IsGGUFPath("./model.gguf"))
	assert.True(t, IsGGUFPath("/tmp/custom-model.gguf"))
	assert.True(t, IsGGUFPath("C:\\models\\test.gguf"))
	assert.False(t, IsGGUFPath("qwen3:8b-q4_0"))
	assert.False(t, IsGGUFPath("llama3"))
}

func TestResolveModelPath_GGUFFile(t *testing.T) {
	dir := t.TempDir()
	ggufPath := filepath.Join(dir, "test.gguf")
	os.WriteFile(ggufPath, []byte("dummy"), 0o644)

	resolved, err := ResolveModelPath(ggufPath)
	assert.NoError(t, err)
	assert.Equal(t, ggufPath, resolved)
}

func TestResolveModelPath_GGUFNotFound(t *testing.T) {
	_, err := ResolveModelPath("/nonexistent/model.gguf")
	assert.Error(t, err)
}
