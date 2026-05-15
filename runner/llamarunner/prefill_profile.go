// Package llamarunner — prefill profiling probes (counterpart to
// ollamarunner/prefill_profile.go).
//
// EXPERIMENT ONLY — opt-in timing for cross-runner comparison. Enable with
// OLLAMA_PREFILL_PROFILE=1.
//
// Unlike the ollama runner, the llama runner delegates virtually all prefill
// work to a single C++ call (lc.Decode). The Go-side wrapping is thin, so
// we only measure two timestamps per batch:
//
//   - Decode(batch) async submit time (this also includes graph build, sched
//     alloc, and kernel launches inside C++)
//   - Synchronize() blocking time (waits for GPU and is required before the
//     logits can be read)
//
// The sum of these two is the closest analog to ollama runner's
// (model_forward + input_inject + compute_outer + floats) — i.e., the
// "equivalent work" we want to compare against.
//
// Probe overhead: 2-3 time.Now() calls per batch + 1 stderr WriteString.
// All probe methods short-circuit on a single bool when disabled.

package llamarunner

import (
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

var prefillProfileEnabled = func() bool {
	v := strings.TrimSpace(os.Getenv("OLLAMA_PREFILL_PROFILE"))
	return v != "" && v != "0" && !strings.EqualFold(v, "false")
}()

// llamaPrefillBatchCounter assigns a monotonically increasing id per profiled
// batch. Used purely for log correlation; no semantic effect on processing.
var llamaPrefillBatchCounter atomic.Int64

func nextLlamaPrefillBatchID() int {
	if !prefillProfileEnabled {
		return 0
	}
	return int(llamaPrefillBatchCounter.Add(1) - 1)
}

// llamaPrefillProfile is the llama-runner counterpart of ollamarunner's
// prefillProfile. Single-goroutine usage (processBatch is sequential), so no
// synchronization is needed for field access.
type llamaPrefillProfile struct {
	enabled bool

	batchID    int
	nTokens    int
	numOutputs int

	tDecodeStart time.Time
	tDecodeEnd   time.Time
	tSyncStart   time.Time
	tSyncEnd     time.Time

	finished int32
}

func newLlamaPrefillProfile(batchID, nTokens, numOutputs int) *llamaPrefillProfile {
	return &llamaPrefillProfile{
		enabled:    prefillProfileEnabled,
		batchID:    batchID,
		nTokens:    nTokens,
		numOutputs: numOutputs,
	}
}

func (p *llamaPrefillProfile) Enabled() bool {
	return p != nil && p.enabled
}

func (p *llamaPrefillProfile) MarkDecode(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tDecodeStart = start
	p.tDecodeEnd = end
}

func (p *llamaPrefillProfile) MarkSync(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tSyncStart = start
	p.tSyncEnd = end
}

// Finish writes a single PREFILL_PROFILE_LLAMA line to stderr.
func (p *llamaPrefillProfile) Finish() {
	if !p.Enabled() {
		return
	}
	if !atomic.CompareAndSwapInt32(&p.finished, 0, 1) {
		return
	}
	if p.nTokens == 0 {
		return
	}

	ms := func(d time.Duration) float64 {
		return float64(d) / float64(time.Millisecond)
	}
	dur := func(start, end time.Time) time.Duration {
		if start.IsZero() || end.IsZero() {
			return 0
		}
		return end.Sub(start)
	}

	decode := dur(p.tDecodeStart, p.tDecodeEnd)
	sync := dur(p.tSyncStart, p.tSyncEnd)
	total := dur(p.tDecodeStart, p.tSyncEnd)
	if total == 0 {
		// Sync wasn't called (numOutputs == 0); fall back to decode only.
		total = decode
	}

	line := fmt.Sprintf(
		"PREFILL_PROFILE_LLAMA batch_id=%d n_tokens=%d n_outputs=%d "+
			"decode=%.3fms sync=%.3fms total=%.3fms\n",
		p.batchID, p.nTokens, p.numOutputs,
		ms(decode), ms(sync), ms(total),
	)

	_, _ = os.Stderr.WriteString(line)
}
