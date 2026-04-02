# bench-sweep Design Spec

**Date**: 2026-04-02  
**Status**: Approved  
**Context**: Intel ARL 265K + 128 GB RAM + RTX 3090 (24 GB), Windows 11, Ollama running Qwen3-coder-next

---

## Background & Motivation

The existing `cmd/bench` tool (from upstream Ollama) can measure prefill TPS, generate TPS, and TTFT for a single prompt size per invocation, but it lacks:

- Multi-size sweep in a single run
- Named run history for cross-run comparison
- Statistical summaries (mean / median / p99 / CV%)
- Automatic stability warnings based on CV%
- A prompt generator accurate enough for large inputs (the existing 62-word list wraps at ~4096+ tokens)

This tool is designed to evaluate the effect of performance optimizations (e.g., CPU affinity, weight swapping, KV cache quantization) by providing reproducible, statistically grounded benchmark runs that can be diff'd against each other.

---

## Binary

**Location**: `cmd/bench-sweep/`  
**Does not modify** `cmd/bench/` — zero coupling to upstream bench code.

### Subcommands

```
bench-sweep run  -model <model> -name <run-name> [-sizes <token-sizes>] [options]
bench-sweep diff <run-a> <run-b>
bench-sweep list
```

---

## Subcommand: `run`

### Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Ollama model name |
| `-name` | (required) | Name for this run, stored as `~/.ollama/bench/<name>.json` |
| `-sizes` | `512,1024,2048,4096` | Comma-separated list of prompt token sizes to sweep |
| `-epochs` | `6` | Number of timed iterations per prompt size |
| `-warmup` | `2` | Warmup iterations before timing begins (not recorded) |
| `-max-tokens` | `16` | Max output tokens per request (keep small to isolate prefill) |
| `-cv-threshold` | `5.0` | CV% above which a result is flagged as unstable |
| `-host` | `http://localhost:11434` | Ollama server address |

### Run flow

For each prompt size in `-sizes`:

1. **Pre-generate prompts** — before any timed requests, generate `warmup + epochs` prompts using the tokenize API (see Prompt Generator section). Cache all prompts in memory.
2. **Warmup phase** — send `warmup` requests, discard results. After the final warmup request, check warmup adequacy (see Stability section).
3. **Timed phase** — send `epochs` requests, recording per-epoch metrics from the Ollama API response plus wall-clock TTFT.
4. **Compute statistics** — calculate mean / median / p99 / stddev / CV% for prefill_tps and TTFT across epochs.
5. **Emit stability warnings** — if CV% exceeds threshold for any metric, print a warning with a hint.
6. **Print per-size result row** to stdout.

After all sizes complete, save the full run to `~/.ollama/bench/<name>.json`.

### Run name conflict handling

If `<name>.json` already exists, the tool auto-renames by appending `_1`, `_2`, etc. until a free name is found, then prints:

```
Warning: run name "baseline" already exists, renamed to "baseline_1"
```

### Metrics collected per epoch

From the Ollama `/api/generate` response:

| Field | Source |
|---|---|
| `prompt_eval_count` | `resp.Metrics.PromptEvalCount` |
| `eval_count` | `resp.Metrics.EvalCount` |
| `prefill_ms` | `resp.Metrics.PromptEvalDuration` (converted to ms) |
| `gen_ms` | `resp.Metrics.EvalDuration` (converted to ms) |
| `ttft_ms` | Wall-clock from request start to first non-empty `resp.Response` token |

Derived per-epoch values:

- `prefill_tps = prompt_eval_count / (prefill_ms / 1000)`
- `gen_tps = eval_count / (gen_ms / 1000)`

### Statistics computed across epochs

For both `prefill_tps` and `ttft_ms`:

- `mean` — arithmetic mean
- `median` — middle value after sorting
- `p99` — 99th percentile; with the default of 6 epochs this equals `max` (meaningful separation requires ~100+ epochs, but max is still a useful outlier bound at small N)
- `stddev` — sample standard deviation
- `cv_pct` — `stddev / mean × 100`

`gen_tps` statistics (mean only) are recorded for completeness but are not primary metrics.

### stdout output

```
Run: baseline  |  Model: qwen3-coder-next  |  2026-04-02 10:00
VRAM used: 22.1 GB / 24 GB  |  Epochs: 6  |  Warmup: 2

prompt_tokens │ prefill_tps (mean) │ prefill_tps (p99) │  CV%  │ TTFT mean │ TTFT p99 │  CV%  │ gen_tps │ status
──────────────┼────────────────────┼───────────────────┼───────┼───────────┼──────────┼───────┼─────────┼────────
          512 │          4,850 t/s │         4,720 t/s │  1.8% │     28 ms │    35 ms │  2.3% │  37 t/s │  ✓
        1,024 │          4,266 t/s │         4,100 t/s │  2.1% │     52 ms │    64 ms │  2.8% │  37 t/s │  ✓
        2,048 │          4,180 t/s │         4,050 t/s │  2.4% │    103 ms │   121 ms │  3.1% │  37 t/s │  ✓
        4,096 │          3,890 t/s │         3,200 t/s │  8.7% │    198 ms │   240 ms │  9.2% │  36 t/s │  ⚠

⚠ [size=4096] prefill_tps CV=8.7% exceeds threshold 5.0%
  hint: consider increasing -warmup (current: 2) or closing background processes
⚠ [size=4096] ttft_ms CV=9.2% exceeds threshold 5.0%
```

---

## Subcommand: `diff`

```
bench-sweep diff <run-a> <run-b>
```

Compares two runs on matching prompt sizes. Sizes present in only one run are skipped with a note.

TTFT improvements (lower is better) are shown as negative `Δ%`. The sign convention is printed in the header.

```
Diff: baseline → cpu-affinity  |  Model: qwen3-coder-next
Note: Δ% negative = improvement for TTFT (lower is better), positive = improvement for prefill_tps (higher is better)

prompt_tokens │ prefill_tps baseline→new   │    Δ%  │ TTFT mean baseline→new │ TTFT p99 baseline→new │    Δ%  │ note
──────────────┼─────────────────────────────┼────────┼───────────────────────┼───────────────────────┼────────┼──────────────────────────
          512 │           4,850 → 5,120 t/s │  +5.6% │         28 → 25 ms    │         35 → 30 ms    │ -14.3% │
        1,024 │           4,266 → 4,580 t/s │  +7.4% │         52 → 47 ms    │         64 → 55 ms    │ -14.1% │
        2,048 │           4,180 → 4,510 t/s │  +7.9% │        103 → 94 ms    │        121 → 108 ms   │ -10.7% │
        4,096 │           3,890 → 4,200 t/s │  +8.0% │        198 → 181 ms   │        240 → 215 ms   │ -10.4% │ ⚠ baseline CV=8.7%
```

The `note` column flags any row where either run has `stable=false` for that size, citing which run and its CV%.

---

## Subcommand: `list`

```
bench-sweep list
```

```
NAME              MODEL                   DATE          SIZES                STABLE
baseline          qwen3-coder-next        2026-04-02    512,1024,2048,4096   3/4
cpu-affinity      qwen3-coder-next        2026-04-02    512,1024,2048,4096   4/4
```

`STABLE` shows how many of the tested sizes had CV% below threshold for both primary metrics.

---

## Prompt Generator

### Goal

Produce `warmup + epochs` text strings per prompt size such that:
- Each string tokenizes to **exactly** `target_tokens` tokens (±1 tolerance)
- No two strings share a common prefix long enough to trigger Ollama's KV cache prefix matching
- Content is realistic English prose (not shuffled word lists)

### Corpus

A ~50 KB block of public-domain English text (Project Gutenberg) is embedded in the binary as a `go:embed` asset. At 50,000 characters, this provides approximately 12,500 tokens. For target sizes requiring more tokens, the corpus is concatenated with itself at different offsets.

### Per-prompt generation algorithm

```
func generatePrompt(corpus string, targetTokens int, epoch int,
                    client *api.Client, model string) (string, int, error):

  1. Compute start offset: offset = (epoch * 7919) % len(corpus)
     (7919 is prime; ensures uniform distribution across epochs)

  2. Extract slice: text = corpus[offset:] + corpus[:offset]
     (wrap-around concatenation to always have enough material)

  3. Initial length estimate: chars = targetTokens * 4
     (conservative: assumes ~4 chars/token for English prose)

  4. Call POST /api/tokenize {model, prompt: text[:chars]}
     to get actual token count

  5. Binary search on chars until tokenize returns targetTokens ± 1

  6. Return trimmed text and actual token count
```

All prompts for a run are pre-generated during a setup phase before any timed request is sent, so tokenize API calls do not affect measurement timing.

### Fallback (if /api/tokenize unavailable)

Use the first warmup request's `prompt_eval_count` as calibration:
- Compute ratio: `ratio = actual / target`
- Adjust all subsequent prompt lengths by `1 / ratio`
- Log a warning that exact token counts are approximate (±5%)

### Large prompt handling

For target sizes where `targetTokens * 4 > len(corpus)`, the corpus is self-concatenated with a different offset per segment to avoid exact repetition. The unique-prefix property is preserved because the epoch offset shifts the entire sequence.

---

## Stability Checks

### Warmup adequacy check

Requires `-warmup >= 2`; skipped silently when `-warmup 1` is set (only one warmup sample, no comparison possible).

After the final warmup request, compare its `prefill_tps` to the first warmup request's `prefill_tps`. If the difference exceeds 15%, the GPU has not yet reached thermal steady state:

```
Warning: warmup may be insufficient — prefill_tps changed 18% between warmup iterations
  hint: increase -warmup (current: 2)
```

### Per-size CV% check

After computing statistics for each size, if `cv_pct > cv_threshold` for **either** `prefill_tps` or `ttft_ms`, flag that size as `stable=false` and print the hint message shown in the stdout output section above.

---

## History File Format

**Location**: `~/.ollama/bench/<run-name>.json`

```json
{
  "name": "baseline",
  "model": "qwen3-coder-next",
  "timestamp": "2026-04-02T10:00:00Z",
  "hardware": {
    "os": "windows",
    "vram_used_bytes": 23757000000,
    "vram_total_bytes": 25769000000
  },
  "config": {
    "epochs": 6,
    "warmup": 2,
    "max_tokens": 16,
    "cv_threshold_pct": 5.0,
    "sizes": [512, 1024, 2048, 4096]
  },
  "results": [
    {
      "prompt_tokens": 512,
      "stable": true,
      "epochs": [
        {
          "prompt_eval_count": 512,
          "eval_count": 16,
          "prefill_ms": 105.4,
          "gen_ms": 432.1,
          "ttft_ms": 112.3,
          "prefill_tps": 4857.0,
          "gen_tps": 37.1
        }
      ],
      "stats": {
        "prefill_tps": { "mean": 4850, "median": 4860, "p99": 4720, "stddev": 87.3, "cv_pct": 1.8 },
        "ttft_ms":     { "mean": 28.1, "median": 27.9, "p99": 35.2, "stddev": 0.65, "cv_pct": 2.3 },
        "gen_tps":     { "mean": 37.2, "median": 37.3, "p99": 36.1, "stddev": 0.4,  "cv_pct": 1.1 }
      }
    }
  ]
}
```

---

## Files to Create

| File | Description |
|---|---|
| `cmd/bench-sweep/main.go` | CLI entry point, flag parsing, subcommand dispatch |
| `cmd/bench-sweep/run.go` | `run` subcommand: sweep loop, warmup, timed epochs, stats |
| `cmd/bench-sweep/diff.go` | `diff` subcommand: load two runs, render comparison table |
| `cmd/bench-sweep/list.go` | `list` subcommand: enumerate and display history |
| `cmd/bench-sweep/prompt.go` | Prompt generator: corpus embed, tokenize API, binary search |
| `cmd/bench-sweep/stats.go` | Statistics: mean, median, p99, stddev, CV% |
| `cmd/bench-sweep/history.go` | JSON read/write, run-name conflict resolution (_1, _2 suffix) |
| `cmd/bench-sweep/corpus.txt` | Embedded ~50 KB public-domain English text |

---

## Runner Path Selection

bench-sweep sends requests through the standard Ollama HTTP API and is agnostic to which runner path is used. The runner is selected by `ollama serve` based on the model and environment, not by bench-sweep.

To benchmark a specific runner, set the environment variable before starting `ollama serve`:

| Env var | Effect |
|---|---|
| (unset, default) | LlamaRunner — vendored llama.cpp via CGO |
| `OLLAMA_NEW_ENGINE=1` | OllamaRunner — Go-native engine; falls back to LlamaRunner automatically if the model is not yet supported (logged as `model not yet supported by Ollama engine, switching to compatibility mode`) |

Qwen3-coder-next is a GGUF model and uses LlamaRunner by default. Whether OllamaRunner supports it depends on the model architecture implementation in `model/models/`.

bench-sweep records `hardware.vram_used_bytes` from `/api/ps` after warmup, which will reflect whichever runner is loaded. Two runs using different runners can be diff'd normally — the runner in use is not captured in the history file, so annotate the run name accordingly (e.g., `baseline-llama` vs `baseline-ollama-engine`).

---

## Non-Goals

- Concurrent request benchmarking (single-user latency focus, not throughput)
- Integration with `trace-analyzer` (future work)
- Accuracy or quality evaluation
