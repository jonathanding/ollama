# bench-sweep

Repeatable inference benchmark for Ollama. Sweeps multiple prompt sizes in one run, measures five metrics (prefill_ms, prefill_tps, ttft_ms, gen_ms, gen_tps), flags unstable results via CV%, and stores named run history for cross-run comparison.

---

## Build from Source

Prerequisites: Go 1.24+, this repo checked out.

**Windows (PowerShell):**
```powershell
go build -o bench-sweep.exe ./cmd/bench-sweep/
```

**Linux / macOS:**
```bash
go build -o bench-sweep ./cmd/bench-sweep/
```

The binary has no runtime dependencies beyond `ollama serve` running.

---

## Prerequisites

Start Ollama with the model you want to benchmark:
```bash
ollama serve
ollama pull qwen3-coder-next
```

To benchmark using the Go-native OllamaRunner engine (if the model supports it):
```powershell
$env:OLLAMA_NEW_ENGINE = "1"
ollama serve
```

---

## Subcommands

### `run` — Execute a benchmark sweep

```bash
bench-sweep run -model <model> -name <run-name> [options]
```

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Ollama model name |
| `-name` | (required) | Run name; auto-renamed `_1`, `_2`… on conflict |
| `-sizes` | `512,1024,2048,4096` | Comma-separated prompt token sizes to sweep |
| `-epochs` | `6` | Timed iterations per size |
| `-warmup` | `4` | Warmup iterations before timing (≥2 recommended) |
| `-max-tokens` | `16` | Max output tokens per request (keep small to isolate prefill) |
| `-cv-threshold` | `5.0` | CV% above which a result is flagged ⚠ unstable |
| `-host` | `$OLLAMA_HOST` | Ollama server URL |

**Example:**
```bash
bench-sweep run -model qwen3-coder-next -name baseline -sizes 512,1024,2048,4096
```

**Output:**
```
Starting benchmark: model=qwen3-coder-next  sizes=512,1024,2048,4096  epochs=6  warmup=4

Model: qwen3-coder-next  |  Epochs: 6  |  Warmup: 4
note: prefill_ms is the server-side prompt processing time (Ollama internal metric); ttft_ms is the wall-clock time to first token (= prefill_ms + HTTP round-trip + scheduling, typically 80–200 ms longer on localhost).

prompt_tokens │ prefill_ms │   CV% │ prefill_tps │   CV% │ ttft_ms    │   CV% │ gen_ms    │   CV% │ gen_tps   │   CV% │ status
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
512           │   1598 ms │  4.6% │     295 t/s │  4.6% │    1688 ms │  3.3% │    842 ms │  1.2% │    19 t/s │  1.2% │ ✓
  note: 1 epoch(s) excluded from stats for size=1024 (early EOS)
1024          │   3140 ms │  2.9% │     308 t/s │  2.9% │    3228 ms │  3.5% │    889 ms │  2.1% │    18 t/s │  2.1% │ ✓
2048          │   6846 ms │  3.7% │     308 t/s │  3.7% │    6946 ms │  5.2% │    941 ms │  1.8% │    17 t/s │  1.8% │ ⚠
⚠ [size=2048] ttft_ms CV=5.2% exceeds threshold 5.0%
  note: 2 epoch(s) excluded from stats for size=4096 (early EOS)
4096          │  12448 ms │  1.0% │     326 t/s │  1.0% │   12548 ms │  2.2% │   1000 ms │  1.4% │    16 t/s │  1.4% │ ✓

Run "baseline" saved to C:\Users\you\.ollama\bench\qwen3-coder-next_baseline.json
```

### `diff` — Compare two runs

```bash
bench-sweep diff <run-a> <run-b>
```

**Example:**
```bash
bench-sweep diff baseline cpu-affinity
```

**Output:**
```
Diff: baseline → cpu-affinity  |  Model: qwen3-coder-next
Note: Δ% negative = improvement for TTFT (lower is better); positive = improvement for prefill_tps (higher is better)

prompt_tokens │ prefill_tps baseline→new      │    Δ%   │ TTFT mean baseline→new  │ TTFT p99 baseline→new  │    Δ%   │ note
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
          512 │    4,850 → 5,120 t/s           │  +5.6%  │      28 → 25 ms         │      35 → 30 ms        │ -14.3%  │
        1,024 │    4,266 → 4,580 t/s           │  +7.4%  │      52 → 47 ms         │      64 → 55 ms        │ -14.1%  │
        4,096 │    3,890 → 4,200 t/s           │  +8.0%  │     198 → 181 ms        │     240 → 215 ms       │ -10.4%  │ ⚠ baseline CV=8.7%
```

### `list` — Show all stored runs

```bash
bench-sweep list
```

**Output:**
```
NAME                     MODEL                        DATE         SIZES                STABLE
────────────────────────────────────────────────────────────────────────────────────────────────
cpu-affinity             qwen3-coder-next             2026-04-02   512,1024,2048,4096   4/4
baseline                 qwen3-coder-next             2026-04-02   512,1024,2048,4096   3/4
```

Run history is stored in `~/.ollama/bench/` as JSON files.

---

## Understanding Results

### prefill_ms
Server-side prompt processing time in milliseconds, derived directly from Ollama's internal `prompt_eval_duration` metric. This is the most stable prefill measurement: it excludes HTTP and scheduling overhead and is measured inside the inference server. Lower is better.

### prefill_tps
Tokens per second during the prefill phase (`prompt_eval_count / prefill_ms × 1000`). Higher is better. This is compute-bound at large prompt sizes: it stresses the model's ability to process input tokens across 512-token batches on the GPU.

`prefill_ms` and `prefill_tps` carry the same information (one is the inverse of the other scaled by token count). `prefill_ms` is more intuitive for latency comparisons; `prefill_tps` is more intuitive for hardware throughput comparisons.

### ttft_ms (Time to First Token)
Wall-clock milliseconds from the moment the request is sent to receipt of the first output token. Equal to `prefill_ms` plus HTTP round-trip and server scheduling overhead — typically **80–200 ms longer** than `prefill_ms` on localhost. Lower is better.

`ttft_ms` directly reflects user-perceived latency; `prefill_ms` isolates the inference engine contribution. Use `prefill_ms` when comparing two runs on the same machine; use `ttft_ms` when reasoning about end-to-end user experience.

### gen_ms
Server-side decode time in milliseconds for the `max_tokens` output tokens, derived from Ollama's internal `eval_duration` metric. With the default `-max-tokens 16`, this is typically 800–1 100 ms depending on model size and hardware.

### gen_tps
Token generation speed during the decode phase (`eval_count / gen_ms × 1000`). Higher is better. This is memory-bandwidth-bound: each generated token requires reading the full model weight from VRAM (or system RAM for CPU-offloaded layers).

> **Note:** With the default `-max-tokens 16`, decode throughput is measured over only 16 tokens. This underestimates steady-state `gen_tps` because per-token CUDA kernel launch overhead is proportionally larger for short sequences. Use `-max-tokens 64` or higher for a more representative decode measurement.

### CV% (Coefficient of Variation)
`stddev / mean × 100`. Measures run-to-run stability. A result is flagged ⚠ if CV% > threshold (default 5%) for either `prefill_tps` or `ttft_ms`. High CV% means the measurement is noisy — a difference between two runs may be noise rather than a real optimisation effect.

### Stability warnings
If CV% exceeds `-cv-threshold` for `prefill_tps` or `ttft_ms` at a given size, that row is marked ⚠. In `diff` output, the note column flags which run was unstable.

**If you see instability warnings:**
1. Increase `-warmup` (try `-warmup 6` or higher; default is already 4)
2. Close browser tabs, background services, and other GPU workloads
3. On Windows, check Task Manager for CPU/GPU spikes during the run

---

## Request Path: `/api/generate` vs `/api/chat`

### How bench-sweep sends requests

Every bench-sweep request uses Ollama's `/api/generate` endpoint with `Raw: true`:

```
POST /api/generate
{ "model": "...", "prompt": "<corpus text>", "raw": true,
  "options": { "temperature": 0, "num_predict": 16 } }
```

`Raw: true` bypasses the chat template — the prompt is sent directly to the model's tokenizer without any special tokens such as `<|im_start|>system`, role markers, or `<think>` delimiters. This is the lowest-overhead path and gives the cleanest measurement of the model's raw prefill and decode throughput.

### Differences from `/api/chat`

| | `/api/generate` + `Raw: true` | `/api/chat` |
|---|---|---|
| Chat template | Bypassed | Applied |
| System prompt | None | Optional |
| Thinking tokens (`<think>…</think>`) | Not triggered | Triggered on models that enable thinking by default |
| TTFT | Time to first answer token | Time to first **visible** answer token (thinking tokens are invisible) |
| Measured generation speed | Pure decode throughput | Decode throughput including hidden thinking tokens |

### Impact on non-thinking models

For models that do not use a thinking mode (e.g. most base models, instruction-tuned models without `<think>` support), there is no practical difference between the two paths except for the chat template token overhead. bench-sweep results are directly comparable to real-world `/api/chat` latency.

### Impact on thinking models (e.g. Qwen3, DeepSeek-R1)

Thinking models use the `<think>…</think>` delimiter in their chat template to activate a chain-of-thought reasoning phase before producing a visible answer. This phase only activates when the template is applied — it is **not** triggered by `Raw: true`.

Consequences for bench-sweep measurements:

- **`prefill_tps` is unaffected.** Prefill (processing the input prompt) is the same regardless of whether thinking is later triggered.
- **`gen_tps` is higher than real-world decode.** With `Raw: true`, the model generates answer tokens directly. In actual `/api/chat` usage the model first generates potentially thousands of invisible thinking tokens, then the answer. The two decode phases have different token distributions, so their per-token throughput differs.
- **`ttft_ms` is much lower than real-world TTFT.** In production, TTFT for a thinking model includes the full thinking phase. A model that thinks for 2 000 tokens before answering will show `TTFT ≈ 2000 / gen_tps` seconds of additional latency that bench-sweep does not capture.

**Example (Qwen3.5 27B, RTX 3090):**

| Metric | bench-sweep (`Raw: true`) | `/api/chat` (thinking) |
|---|---|---|
| prefill_tps | ~470 t/s | ~470 t/s |
| gen_tps | ~24 t/s | ~36 t/s\* |
| user-perceived TTFT (1 K thinking tokens) | ~35 ms | ~35 ms + ~28 s |

\* Thinking-token generation is faster than answer-token generation on Qwen3 because the model's internal routing favours shorter attention spans during the reasoning phase.

### When bench-sweep is the right tool

bench-sweep is well-suited for:

- **Comparing optimisations on the same model** — any change that affects prefill or decode throughput is correctly captured regardless of endpoint.
- **Non-thinking models** — results directly reflect user-facing latency.
- **Thinking models where prefill performance is the focus** — prefill_tps and TTFT attributable to the prefill phase are accurately measured.

For thinking models where **end-to-end user-perceived TTFT** matters, a complementary benchmark using `/api/chat` (with the chat template applied and thinking enabled) is needed. The invisible thinking phase dominates TTFT for complex prompts and is outside bench-sweep's current scope.

---

## Runner Path Selection

bench-sweep sends requests through the standard Ollama HTTP API and does not control which runner is used. Set the environment variable **before** starting `ollama serve`:

| Env var | Effect |
|---|---|
| (unset, default) | LlamaRunner — vendored llama.cpp via CGO |
| `OLLAMA_NEW_ENGINE=1` | OllamaRunner — Go-native engine; falls back to LlamaRunner if the model is not yet supported |

When comparing LlamaRunner vs OllamaRunner, name your runs accordingly (e.g., `baseline-llama` vs `baseline-ollama-engine`) since the runner in use is not recorded in the history file.

---

## Design Notes

**Why a separate tool from `cmd/bench`?**
`cmd/bench` is upstream Ollama code kept in sync with the main project. `bench-sweep` adds opinionated features (multi-size sweep, history, CV% checks) without modifying upstream code.

**Why use `prompt_eval_duration` rather than wall-clock prefill time?**
`prompt_eval_duration` is measured inside the Ollama server, excluding HTTP and scheduling overhead. It is more stable across runs. TTFT (wall-clock) is still measured separately for user-experience context.

**Why only 16 output tokens by default?**
The goal is to stress prefill. Fewer output tokens means less time in the decode phase, making the benchmark faster and keeping prefill as the dominant cost.

**Why vary the prompt per epoch?**
Ollama caches KV state for prompts with matching prefixes. Without variation, every epoch after the first would get a cache hit and show unrealistically fast prefill. The corpus-based generator ensures each epoch starts at a different offset in a 50 KB public-domain text.

**How is the prompt token count calibrated?**
The first warmup request is sent with a heuristic prompt length (4 chars/token). The actual `prompt_eval_count` from the response is used to compute a scaling factor applied to all subsequent requests for that size. Accuracy is typically ±5 tokens.
