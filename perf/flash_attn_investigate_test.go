package perf

import (
	"math"
	"testing"

	"github.com/ollama/ollama/ml"
	_ "github.com/ollama/ollama/ml/backend"
)

// TestFlashAttnInvestigate measures FLASH_ATTN_EXT under 4 conditions to isolate
// why the benchmark underestimates actual inference by ~12x.
//
// Actual inference at input_length=300: q(128,325,16,1) k(128,512,8,1) → ~58K us per layer
// Benchmark profile at num_heads=16, (464,464): ~5K us
//
// Run: DAOP_INTEGRATION=1 go test ./perf/ -run TestFlashAttnInvestigate -v -count=1
func TestFlashAttnInvestigate(t *testing.T) {
	backend := skipIfNoBackend(t)
	defer backend.Close()

	caps := DiscoverBackend(backend)
	t.Logf("Backend: %s, HasGPUTimestamp: %v", caps.Name, caps.HasGPUTimestamp)

	const (
		headDim    = 128
		seqQ       = 325
		seqKV      = 512
		numQHeads  = 16
		numKVHeads = 8
	)

	cfg := BenchmarkConfig{WarmupReps: 3, MeasureReps: 5, TrimPercent: 0.1}

	// Helper: create causal mask [seqKV, seqQ] where mask[kv,q] = 0 if kv<=q, else -inf
	makeCausalMaskF32 := func(ctx ml.Context) ml.Tensor {
		data := make([]float32, seqKV*seqQ)
		for q := 0; q < seqQ; q++ {
			for kv := 0; kv < seqKV; kv++ {
				if kv > q {
					data[q*seqKV+kv] = float32(math.Inf(-1))
				}
			}
		}
		return ctx.Input().FromFloats(data, seqKV, seqQ)
	}

	type experiment struct {
		name    string
		qHeads  int
		kvHeads int
		useMask bool
	}

	experiments := []experiment{
		{"baseline (Q=16, KV=16, no mask)", numQHeads, numQHeads, false},
		{"GQA only (Q=16, KV=8, no mask)", numQHeads, numKVHeads, false},
		{"mask only (Q=16, KV=16, causal)", numQHeads, numQHeads, true},
		{"GQA+mask (Q=16, KV=8, causal)", numQHeads, numKVHeads, true},
	}

	for _, exp := range experiments {
		qH := exp.qHeads
		kvH := exp.kvHeads
		wantMask := exp.useMask

		// Register temporary op
		opName := "FLASH_ATTN_INVESTIGATE"
		opRegistry[opName] = OpRunnerML{
			Dimensions: []string{"seq_q", "seq_kv"},
			CreateInputs: func(ctx ml.Context, be ml.Backend, _ string, _ []int64) []ml.Tensor {
				q := randomTensor(ctx, ml.DTypeF32, headDim, qH, seqQ, 1)
				kBytes := materializeTensor(be, ml.DTypeF16, headDim, kvH, seqKV, 1)
				vBytes := materializeTensor(be, ml.DTypeF16, headDim, kvH, seqKV, 1)
				k := ctx.Input().FromBytes(ml.DTypeF16, kBytes, headDim, kvH, seqKV, 1)
				v := ctx.Input().FromBytes(ml.DTypeF16, vBytes, headDim, kvH, seqKV, 1)
				tensors := []ml.Tensor{q, k, v}
				if wantMask {
					tensors = append(tensors, makeCausalMaskF32(ctx))
				}
				return tensors
			},
			Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
				sdpa, ok := in[0].(ml.ScaledDotProductAttention)
				if !ok {
					return nil
				}
				scale := 1.0 / math.Sqrt(float64(headDim))
				var mask ml.Tensor
				if len(in) > 3 {
					mask = in[3]
				}
				// Pass cacheConfigApplied=true to skip mask dtype conversion
				// (benchmark backend may not have flash attention cache config initialized)
				return sdpa.ScaledDotProductAttention(ctx, in[1], in[2], mask, nil, nil, scale, true)
			},
		}

		pt := measureOpForBackend(backend, caps, opName, []int64{int64(seqQ), int64(seqKV)}, "f16", cfg)
		t.Logf("%-40s → %10.1f us (stddev: %.1f, reps: %d)",
			exp.name, pt.LatencyUs, pt.StddevUs, pt.Reps)

		delete(opRegistry, opName)
	}

	t.Log("")
	t.Logf("Reference: actual inference per-layer FLASH_ATTN = ~58,356 us")
	t.Logf("Reference: profile curve num_heads=16 at (464,464) = ~4,942 us")
	t.Logf("Reference: profile curve num_heads=16 at (283,283) = ~2,333 us")
}
