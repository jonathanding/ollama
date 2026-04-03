package perf

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed viewer.html
var htmlTemplate string

// GenerateHTMLViewer creates a self-contained HTML file with interactive charts.
func GenerateHTMLViewer(profile *Profile, outputPath string) error {
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		return fmt.Errorf("marshal profile: %w", err)
	}

	html := strings.Replace(htmlTemplate, "__PROFILE_JSON__", string(profileJSON), 1)

	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}

	return os.WriteFile(outputPath, []byte(html), 0o644)
}

// OpenHTMLViewer opens the HTML file in the default browser.
func OpenHTMLViewer(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}

	fmt.Printf("Open in browser: file://%s\n", filepath.ToSlash(absPath))
	return nil
}
