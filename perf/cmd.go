package perf

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"strings"

	"github.com/ollama/ollama/ml"
)

// RunBenchmarkCLI is the entry point for `ollama daop-bench`.
func RunBenchmarkCLI(backend ml.Backend, opts BenchmarkCLIOptions) error {
	cfg := DefaultBenchmarkConfig()

	ops := []string{"SILU", "MUL_MAT", "FLASH_ATTN_EXT"}
	if opts.Ops != "" {
		ops = strings.Split(opts.Ops, ",")
	}

	dtypes := Phase1Dtypes()
	if opts.Dtypes != "" {
		dtypes = strings.Split(opts.Dtypes, ",")
	}

	slog.Info("starting calibration", "ops", ops, "dtypes", dtypes)

	profile, err := RunBenchmark(backend, ops, dtypes, cfg)
	if err != nil {
		return fmt.Errorf("benchmark failed: %w", err)
	}

	outputPath := opts.Output
	if outputPath == "" {
		outputPath = ProfilePath()
	}

	if err := WriteProfile(outputPath, profile); err != nil {
		return fmt.Errorf("write profile: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Profile saved to %s\n", outputPath)

	if opts.Verbose {
		PrintProfile(os.Stdout, profile, true)
	} else {
		PrintProfile(os.Stdout, profile, false)
	}

	if opts.Viewer {
		htmlPath := outputPath + ".html"
		if err := GenerateHTMLViewer(profile, htmlPath); err != nil {
			return fmt.Errorf("generate viewer: %w", err)
		}
		fmt.Fprintf(os.Stderr, "HTML viewer saved to %s\n", htmlPath)
		openBrowser(htmlPath)
	}

	return nil
}

// RunEstimateCLI is the entry point for `ollama daop-estimate`.
func RunEstimateCLI(modelRef string, opts EstimateCLIOptions) error {
	result, err := RunEstimate(modelRef, opts.Profile)
	if err != nil {
		return err
	}

	if opts.JSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	PrintEstimateResult(os.Stdout, result, opts.Verbose)
	return nil
}

// RunViewerCLI is the entry point for `ollama daop-viewer`.
func RunViewerCLI(opts ViewerCLIOptions) error {
	profilePath := opts.Profile
	if profilePath == "" {
		profilePath = ProfilePath()
	}

	profile, err := LoadProfile(profilePath)
	if err != nil {
		return fmt.Errorf("load profile: %w (have you run 'ollama daop-bench'?)", err)
	}

	outputPath := opts.Output
	if outputPath == "" {
		outputPath = profilePath + ".html"
	}

	if err := GenerateHTMLViewer(profile, outputPath); err != nil {
		return fmt.Errorf("generate viewer: %w", err)
	}

	fmt.Fprintf(os.Stderr, "HTML viewer saved to %s\n", outputPath)
	if opts.Output == "" {
		openBrowser(outputPath)
	}
	return nil
}

// BenchmarkCLIOptions controls `ollama daop-bench`.
type BenchmarkCLIOptions struct {
	Output  string // --output: profile output path
	Ops     string // --ops: comma-separated op list
	Dtypes  string // --dtypes: comma-separated dtype list
	Viewer  bool   // --viewer: generate HTML viewer after benchmarking
	Verbose bool   // --verbose: show per-point results
}

// EstimateCLIOptions controls `ollama daop-estimate`.
type EstimateCLIOptions struct {
	Profile string // --profile: profile path
	JSON    bool   // --json: output as JSON
	Verbose bool   // --verbose: show per-op breakdown
}

// ViewerCLIOptions controls `ollama daop-viewer`.
type ViewerCLIOptions struct {
	Profile string // --profile: profile path
	Output  string // --output: save HTML instead of opening browser
}

// openBrowser opens a file in the system default browser.
func openBrowser(path string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", path)
	case "linux":
		cmd = exec.Command("xdg-open", path)
	case "windows":
		cmd = exec.Command("cmd", "/c", "start", "", path)
	default:
		fmt.Fprintf(os.Stderr, "Open in browser: %s\n", path)
		return
	}
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Could not open browser: %v\nOpen manually: %s\n", err, path)
	}
}
