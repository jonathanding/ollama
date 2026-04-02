package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = runRun(os.Args[2:])
	case "diff":
		err = runDiff(os.Args[2:])
	case "list":
		err = runList(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown subcommand %q\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintln(os.Stderr, "bench-sweep — Ollama inference benchmark with multi-size sweep and run history")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  bench-sweep run  -model <model> -name <run-name> [options]")
	fmt.Fprintln(os.Stderr, "  bench-sweep diff <run-a> <run-b>")
	fmt.Fprintln(os.Stderr, "  bench-sweep list")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Run 'bench-sweep run -help' for run options.")
}

func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	model     := fs.String("model", "", "Ollama model name (required)")
	name      := fs.String("name", "", "Run name for history (required)")
	sizesStr  := fs.String("sizes", "512,1024,2048,4096", "Comma-separated prompt token sizes to sweep")
	epochs    := fs.Int("epochs", 6, "Number of timed iterations per size")
	warmup    := fs.Int("warmup", 2, "Warmup iterations before timing (>=2 recommended)")
	maxTokens := fs.Int("max-tokens", 16, "Max output tokens per request")
	cvThresh  := fs.Float64("cv-threshold", 5.0, "CV% threshold above which a result is flagged unstable")
	host      := fs.String("host", "", "Ollama host URL (default: $OLLAMA_HOST or http://localhost:11434)")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		return fmt.Errorf("-model is required")
	}
	if *name == "" {
		return fmt.Errorf("-name is required")
	}

	var sizes []int
	for _, s := range strings.Split(*sizesStr, ",") {
		s = strings.TrimSpace(s)
		n, err := strconv.Atoi(s)
		if err != nil || n <= 0 {
			return fmt.Errorf("invalid size %q: must be a positive integer", s)
		}
		sizes = append(sizes, n)
	}

	if *host != "" {
		os.Setenv("OLLAMA_HOST", *host)
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("cannot create Ollama client: %w", err)
	}

	cfg := RunConfig{
		Epochs:      *epochs,
		Warmup:      *warmup,
		MaxTokens:   *maxTokens,
		CVThreshPct: *cvThresh,
		Sizes:       sizes,
	}

	dir, err := historyDir()
	if err != nil {
		return err
	}
	chosen, renamed := resolveRunName(dir, *name)
	if renamed {
		fmt.Fprintf(os.Stderr, "Warning: run name %q already exists, renamed to %q\n", *name, chosen)
	}

	fmt.Printf("Starting benchmark: model=%s  sizes=%s  epochs=%d  warmup=%d\n",
		*model, *sizesStr, *epochs, *warmup)

	sizeResults, hw, err := runBenchmark(context.Background(), client, *model, cfg)
	if err != nil {
		return err
	}

	rec := RunRecord{
		Name:      chosen,
		Model:     *model,
		Timestamp: time.Now().UTC(),
		Hardware:  hw,
		Config:    cfg,
		Results:   sizeResults,
	}
	if err := saveRun(rec); err != nil {
		return fmt.Errorf("save run: %w", err)
	}
	fmt.Printf("\nRun %q saved to %s\n", chosen, filepath.Join(dir, chosen+".json"))
	return nil
}
