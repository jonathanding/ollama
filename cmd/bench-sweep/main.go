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
	case "help":
		printHelp()
		return
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
	fmt.Fprintln(os.Stderr, "  bench-sweep help")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Run 'bench-sweep help' for detailed usage, or 'bench-sweep run -help' for run options.")
}

func printHelp() {
	fmt.Print(`bench-sweep — Ollama inference benchmark with multi-size sweep and run history

USAGE
  bench-sweep run  -model <model> -name <run-name> [options]
  bench-sweep diff <run-a> <run-b>
  bench-sweep list
  bench-sweep help

SUBCOMMANDS
  run     Run a benchmark sweep across multiple prompt sizes.
          Results are saved to ~/.ollama/bench/<model>_<name>.json.

  diff    Compare two saved runs side-by-side.
          Shows Δ% for prefill_tps and TTFT mean/p99.
          Negative Δ% for TTFT = improvement (lower is better).
          Rows where either run has CV% above threshold are flagged ⚠.

  list    List all saved runs with model, date, sizes, and stability summary.

  help    Show this message.

RUN FLAGS
  -model <name>         Ollama model name (required)
  -name <name>          Run name stored as ~/.ollama/bench/<model>_<name>.json (required)
                        If the name already exists it is auto-renamed (_1, _2, …)
  -sizes <list>         Comma-separated prompt token sizes to sweep
                        Default: 512,1024,2048,4096
  -epochs <n>           Timed iterations per size  (default: 6)
  -warmup <n>           Warmup iterations before timing, discarded  (default: 4)
  -max-tokens <n>       Max output tokens per request  (default: 16)
                        Keep small to isolate prefill; output is not the focus.
  -cv-threshold <pct>   CV% above which a size is flagged unstable  (default: 5.0)
  -num-ctx <n>          KV cache context size passed to Ollama as num_ctx  (default: 0 = follow model Modelfile)
                        Smaller values reduce VRAM usage; may allow more model layers to stay on GPU.
  -batch-size <n>       Prompt processing batch size passed as num_batch  (default: 0 = Ollama default, typically 512)
                        Larger values improve GPU utilization during prefill at the cost of peak memory.
  -host <url>           Ollama server URL  (default: $OLLAMA_HOST or http://localhost:11434)

METRICS
  prefill_tps   Tokens processed per second during the prompt-eval (prefill) phase.
                Derived from Ollama's server-side PromptEvalDuration.
  ttft_ms       Time to First Token — wall-clock milliseconds from request send
                to receipt of the first non-empty response token.
                Primary latency signal; directly tied to prefill speed and user experience.
  gen_tps       Generation tokens per second (recorded but not a primary metric).
  CV%           Coefficient of Variation = stddev/mean × 100.
                Measures result stability. Values above -cv-threshold are flagged ⚠.
  p99           99th percentile across epochs.
                With the default 6 epochs this equals the max value.

PROMPT GENERATION
  bench-sweep builds prompts from an embedded public-domain English corpus.
  Each epoch uses a unique corpus offset (based on a prime stride) to defeat
  Ollama's KV-cache prefix matching and ensure independent measurements.
  Token count is calibrated from the first warmup request's prompt_eval_count,
  then scaled for subsequent requests (±5% accuracy without /api/tokenize).

STABILITY WARNINGS
  Warmup adequacy: if prefill_tps changed >15% between the first and last warmup
  iteration, the GPU may not be at thermal steady state. Increase -warmup.

  Per-size CV%: if CV% exceeds -cv-threshold for prefill_tps or ttft_ms,
  the size is marked ⚠ in the table and a hint is printed.

RUNNER PATH
  bench-sweep is agnostic to the Ollama runner. The runner is chosen by
  'ollama serve' based on environment:
    (unset)              LlamaRunner — vendored llama.cpp via CGO  (default)
    OLLAMA_NEW_ENGINE=1  OllamaRunner — Go-native engine

  To compare runners, start 'ollama serve' with the desired env var and name
  your runs accordingly, e.g. "baseline-llama" vs "baseline-ollama-engine".

EXAMPLES
  # Run a baseline sweep
  bench-sweep run -model qwen3-coder-next -name baseline

  # Run with more epochs for tighter statistics
  bench-sweep run -model qwen3-coder-next -name stable -epochs 12 -warmup 4

  # Sweep only large sizes
  bench-sweep run -model qwen3-coder-next -name large -sizes 4096,8192,16384

  # Compare two runs
  bench-sweep diff baseline cpu-affinity

  # List all saved runs
  bench-sweep list

HISTORY FILES
  Runs are stored as JSON in ~/.ollama/bench/<model>_<name>.json and contain
  the full per-epoch measurements, statistics, hardware snapshot, and run config.
  The model name is sanitized (colons and slashes replaced with underscores,
  ":latest" suffix stripped) to produce a valid filename component.
`)
}

func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	model     := fs.String("model", "", "Ollama model name (required)")
	name      := fs.String("name", "", "Run name for history (required)")
	sizesStr  := fs.String("sizes", "512,1024,2048,4096", "Comma-separated prompt token sizes to sweep")
	epochs    := fs.Int("epochs", 6, "Number of timed iterations per size")
	warmup    := fs.Int("warmup", 4, "Warmup iterations before timing (>=2 recommended)")
	maxTokens := fs.Int("max-tokens", 16, "Max output tokens per request")
	cvThresh  := fs.Float64("cv-threshold", 5.0, "CV% threshold above which a result is flagged unstable")
	numCtx    := fs.Int("num-ctx", 0, "KV cache context size (0 = follow model Modelfile default)")
	batchSize := fs.Int("batch-size", 0, "Prompt processing batch size / n_batch (0 = use Ollama default, typically 512)")
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
		NumCtx:      *numCtx,
		BatchSize:   *batchSize,
	}

	dir, err := historyDir()
	if err != nil {
		return err
	}
	chosen, renamed := resolveRunName(dir, *model, *name)
	if renamed {
		fmt.Fprintf(os.Stderr, "Warning: run name %q already exists, renamed to %q\n", *name, chosen)
	}

	startMsg := fmt.Sprintf("Starting benchmark: model=%s  sizes=%s  epochs=%d  warmup=%d", *model, *sizesStr, *epochs, *warmup)
	if *numCtx > 0 {
		startMsg += fmt.Sprintf("  num-ctx=%d", *numCtx)
	}
	if *batchSize > 0 {
		startMsg += fmt.Sprintf("  batch-size=%d", *batchSize)
	}
	fmt.Println(startMsg)

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
	fmt.Printf("\nRun %q saved to %s\n", chosen, filepath.Join(dir, runFileName(*model, chosen)+".json"))
	return nil
}
