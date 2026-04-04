package perf

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateHTMLViewer_ProducesValidHTML(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	html := string(data)

	assert.Contains(t, html, "<!DOCTYPE html>")
	assert.Contains(t, html, "</html>")
	assert.Contains(t, html, "plotly")
	assert.Contains(t, html, "PROFILE_DATA")
}

func TestGenerateHTMLViewer_ContainsProfileData(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	html := string(data)

	assert.Contains(t, html, "SILU")
	assert.Contains(t, html, "MUL_MAT")
}

func TestGenerateHTMLViewer_ContainsChartElements(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	html := string(data)

	assert.Contains(t, html, "charts-container")
	assert.Contains(t, html, "chart-card")
}

func TestGenerateHTMLViewer_EmptyProfile(t *testing.T) {
	p := &Profile{Version: 2}
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.True(t, len(data) > 0)
}

func TestGenerateHTMLViewer_FileSize(t *testing.T) {
	p := newTestProfile()
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	info, err := os.Stat(path)
	require.NoError(t, err)
	assert.Greater(t, info.Size(), int64(1000))
	assert.Less(t, info.Size(), int64(1000000))
}

func TestGenerateHTMLViewer_SpecialCharsInData(t *testing.T) {
	p := &Profile{
		Version: 2,
		Hardware: HardwareProfile{
			Backends: []BackendInfo{{Name: "cuda", Device: "NVIDIA RTX 4090 <test>"}},
		},
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "viewer.html")

	err := GenerateHTMLViewer(p, path)
	require.NoError(t, err)

	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.False(t, strings.Contains(string(data), "<test>"),
		"device name should be JSON-escaped, not raw HTML")
}
