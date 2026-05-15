// Package ollamarunner — prefill profiling probes.
//
// EXPERIMENT ONLY — opt-in timing probes for diagnosing the prefill performance
// gap between ollamarunner and llamarunner on hybrid models such as
// qwen3-coder-next. Enable with OLLAMA_PREFILL_PROFILE=1.
//
// Design constraints (must be preserved):
//
//   1. forwardBatch and computeBatch run in separate goroutines; the profile
//      struct lives on batchState so the existing channel handoffs
//      (computeStartedCh / inputsReadyCh / outputsReadyCh) provide
//      happens-before for all profile field writes — no extra locks needed.
//   2. The ggml backend's ComputeWithNotify spawns `go cb()` to signal the
//      next forwardBatch to start. Probes must NOT delay the cb() goroutine
//      or hold schedMu any longer than the original code did. We achieve this
//      by writing timestamps directly into pre-existing fields (no allocation,
//      no channel ops, no syscalls) inside the schedMu critical section.
//   3. Floats() lazy-syncs the GPU. We measure it from computeBatch (the
//      caller side), not by injecting probes into ml.Context.
//   4. Finish() writes to stderr — call it only AFTER all channel sends and
//      lock releases in computeBatch, so the syscall does not stall the next
//      batch's forwardBatch.
//
// When OLLAMA_PREFILL_PROFILE is unset, all probe methods short-circuit on
// a single bool check; the per-batch overhead is bounded by ~14 time.Now()
// calls (≈1µs total on Windows), well below measurement noise.

package ollamarunner

import (
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

// prefillProfileEnabled is read once at process start so hot paths can
// short-circuit with a single load.
var prefillProfileEnabled = func() bool {
	v := strings.TrimSpace(os.Getenv("OLLAMA_PREFILL_PROFILE"))
	return v != "" && v != "0" && !strings.EqualFold(v, "false")
}()

// prefillProfile collects timing samples for a single batch. It is allocated
// once per batch and stored on batchState so the existing channel handoffs
// between forwardBatch and computeBatch already provide a happens-before
// relationship for all field writes/reads. Do not introduce additional locks.
//
// The struct is intentionally small and POD-like — passing it across the
// channel pipeline must not require synchronization beyond what batchState
// already does.
type prefillProfile struct {
	enabled bool // immutable after construction

	batchID  int
	nInputs  int
	nOutputs int

	// forwardBatch goroutine writes these.
	tForwardStart      time.Time
	tNewContextStart   time.Time
	tNewContextEnd     time.Time
	tModelForwardStart time.Time
	tModelForwardEnd   time.Time
	tForwardEnd        time.Time

	// computeBatch goroutine writes these.
	tComputeBatchStart time.Time
	tInputInjectStart  time.Time
	tInputInjectEnd    time.Time
	tComputeOuterStart time.Time
	tComputeOuterEnd   time.Time
	tFloatsStart       time.Time
	tFloatsEnd         time.Time
	tCloseStart        time.Time
	tCloseEnd          time.Time

	// Backend-side breakdown of ComputeWithNotify, written from inside the
	// ml backend (ggml.go) while holding schedMu. Reads happen on the same
	// goroutine that wrote them, so no synchronization is needed.
	dSchedComputeAsync time.Duration
	dSchedReset        time.Duration

	// dSchedSync is written when Floats()/sync() actually blocks; reader is
	// the same goroutine.
	dSchedSync time.Duration

	// Graph statistics, populated by the ml backend at compute time.
	graphNodes  int
	graphSplits int

	// finished guards Finish() against double-emit.
	finished int32
}

// newPrefillProfile returns a profile that is no-op when env switch is off.
// The returned pointer is always non-nil so callers don't need nil checks.
func newPrefillProfile(batchID int) *prefillProfile {
	return &prefillProfile{enabled: prefillProfileEnabled, batchID: batchID}
}

func (p *prefillProfile) Enabled() bool {
	return p != nil && p.enabled
}

func (p *prefillProfile) MarkForwardStart() {
	if !p.Enabled() {
		return
	}
	p.tForwardStart = time.Now()
}

func (p *prefillProfile) MarkNewContext(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tNewContextStart = start
	p.tNewContextEnd = end
}

func (p *prefillProfile) MarkModelForward(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tModelForwardStart = start
	p.tModelForwardEnd = end
}

func (p *prefillProfile) MarkForwardEnd(nInputs, nOutputs int) {
	if !p.Enabled() {
		return
	}
	p.tForwardEnd = time.Now()
	p.nInputs = nInputs
	p.nOutputs = nOutputs
}

func (p *prefillProfile) MarkComputeBatchStart() {
	if !p.Enabled() {
		return
	}
	p.tComputeBatchStart = time.Now()
}

func (p *prefillProfile) MarkInputInject(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tInputInjectStart = start
	p.tInputInjectEnd = end
}

func (p *prefillProfile) MarkComputeOuter(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tComputeOuterStart = start
	p.tComputeOuterEnd = end
}

func (p *prefillProfile) MarkFloats(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tFloatsStart = start
	p.tFloatsEnd = end
}

func (p *prefillProfile) MarkClose(start, end time.Time) {
	if !p.Enabled() {
		return
	}
	p.tCloseStart = start
	p.tCloseEnd = end
}

// SetComputeBreakdown records sub-stage durations measured inside ggml's
// ComputeWithNotify while schedMu is held. Called from the ml backend.
func (p *prefillProfile) SetComputeBreakdown(schedComputeAsync, schedReset time.Duration) {
	if !p.Enabled() {
		return
	}
	p.dSchedComputeAsync = schedComputeAsync
	p.dSchedReset = schedReset
}

// SetSyncDuration records how long the Floats()/sync() actually blocked.
// Called from the ml backend.
func (p *prefillProfile) SetSyncDuration(d time.Duration) {
	if !p.Enabled() {
		return
	}
	p.dSchedSync = d
}

// SetGraphStats records graph node and split counts.
func (p *prefillProfile) SetGraphStats(nodes, splits int) {
	if !p.Enabled() {
		return
	}
	p.graphNodes = nodes
	p.graphSplits = splits
}

// Finish writes a single structured PREFILL_PROFILE line to stderr. It must be
// called from computeBatch AFTER outputsReadyCh has been signaled and after
// ctx.Close() — otherwise the stderr syscall would stall pipelined work. It is
// idempotent and skips emission for empty (idle) batches.
func (p *prefillProfile) Finish() {
	if !p.Enabled() {
		return
	}
	if !atomic.CompareAndSwapInt32(&p.finished, 0, 1) {
		return
	}
	if p.nInputs == 0 {
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

	forwardTotal := dur(p.tForwardStart, p.tForwardEnd)
	newCtx := dur(p.tNewContextStart, p.tNewContextEnd)
	modelFwd := dur(p.tModelForwardStart, p.tModelForwardEnd)
	forwardOther := forwardTotal - newCtx - modelFwd
	if forwardOther < 0 {
		forwardOther = 0
	}

	inputInject := dur(p.tInputInjectStart, p.tInputInjectEnd)
	computeOuter := dur(p.tComputeOuterStart, p.tComputeOuterEnd)
	floats := dur(p.tFloatsStart, p.tFloatsEnd)
	closeDur := dur(p.tCloseStart, p.tCloseEnd)
	computeBatchTotal := dur(p.tComputeBatchStart, p.tCloseEnd)

	// Single-line key=value format. One atomic write to keep the line
	// uninterleaved across goroutines.
	line := fmt.Sprintf(
		"PREFILL_PROFILE batch_id=%d n_inputs=%d n_outputs=%d "+
			"forward_total=%.3fms new_context=%.3fms model_forward=%.3fms forward_other=%.3fms "+
			"compute_total=%.3fms input_inject=%.3fms compute_outer=%.3fms "+
			"sched_compute_async=%.3fms sched_reset=%.3fms sched_sync=%.3fms "+
			"floats=%.3fms close=%.3fms "+
			"graph_nodes=%d graph_splits=%d\n",
		p.batchID, p.nInputs, p.nOutputs,
		ms(forwardTotal), ms(newCtx), ms(modelFwd), ms(forwardOther),
		ms(computeBatchTotal), ms(inputInject), ms(computeOuter),
		ms(p.dSchedComputeAsync), ms(p.dSchedReset), ms(p.dSchedSync),
		ms(floats), ms(closeDur),
		p.graphNodes, p.graphSplits,
	)

	_, _ = os.Stderr.WriteString(line)
}
