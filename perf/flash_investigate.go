package perf

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
)

// RunFlashAttnInvestigate measures FLASH_ATTN_EXT under multiple conditions to isolate
// why the benchmark underestimates actual inference by ~12x.
func RunFlashAttnInvestigate(backend ml.Backend) error {
	caps := DiscoverBackend(backend)
	fmt.Printf("Backend: %s, HasGPUTimestamp: %v\n", caps.Name, caps.HasGPUTimestamp)

	const (
		headDim    = 128
		seqQ       = 325
		seqKV      = 512
		numQHeads  = 16
		numKVHeads = 8
	)

	cfg := BenchmarkConfig{WarmupReps: 3, MeasureReps: 5, TrimPercent: 0.1}

	type experiment struct {
		name    string
		qHeads  int
		kvHeads int
		useMask bool
	}

	experiments := []experiment{
		{"baseline (Q=16, KV=16, no mask)", numQHeads, numQHeads, false},
		{"GQA only (Q=16, KV=8, no mask)", numQHeads, numKVHeads, false},
	}

	fmt.Printf("\nFLASH_ATTN_EXT investigation: seqQ=%d, seqKV=%d, headDim=%d\n\n", seqQ, seqKV, headDim)
	fmt.Println("=== Non-flash path (decomposed MulmatFullPrec + softmax + Mulmat) ===")

	for _, exp := range experiments {
		qH := exp.qHeads
		kvH := exp.kvHeads
		wantMask := exp.useMask

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
					tensors = append(tensors, makeCausalMaskF32(ctx, seqKV, seqQ))
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
				return sdpa.ScaledDotProductAttention(ctx, in[1], in[2], mask, nil, nil, scale, false)
			},
		}

		pt := measureOpForBackend(backend, caps, opName, []int64{int64(seqQ), int64(seqKV)}, "f16", cfg)
		fmt.Printf("  %-40s → %10.1f us (stddev: %.1f, reps: %d)\n",
			exp.name, pt.LatencyUs, pt.StddevUs, pt.Reps)

		delete(opRegistry, opName)
	}

	fmt.Println()
	fmt.Println("Reference: actual inference per-layer FLASH_ATTN = ~58,356 us")
	fmt.Println("Reference: profile curve num_heads=16 at (464,464) = ~4,942 us")
	fmt.Println("Reference: profile curve num_heads=16 at (283,283) = ~2,333 us")

	return nil
}

// RunFlashAttnInvestigateFlash measures with flash attention enabled (fused kernel).
func RunFlashAttnInvestigateFlash(backend ml.Backend) error {
	caps := DiscoverBackend(backend)
	fmt.Printf("Backend: %s, HasGPUTimestamp: %v\n", caps.Name, caps.HasGPUTimestamp)

	const (
		headDim    = 128
		seqQ       = 325
		seqKV      = 512
		numQHeads  = 16
		numKVHeads = 8
	)

	cfg := BenchmarkConfig{WarmupReps: 3, MeasureReps: 5, TrimPercent: 0.1}

	fmt.Printf("\nFLASH_ATTN_EXT investigation (FLASH ENABLED): seqQ=%d, seqKV=%d, headDim=%d\n\n", seqQ, seqKV, headDim)
	fmt.Println("=== Flash attention path (fused ggml_flash_attn_ext kernel) ===")

	type experiment struct {
		name    string
		qHeads  int
		kvHeads int
	}

	experiments := []experiment{
		{"flash baseline (Q=16, KV=16)", numQHeads, numQHeads},
		{"flash GQA (Q=16, KV=8)", numQHeads, numKVHeads},
	}

	for _, exp := range experiments {
		qH := exp.qHeads
		kvH := exp.kvHeads

		opName := "FLASH_ATTN_INVESTIGATE_FLASH"
		opRegistry[opName] = OpRunnerML{
			Dimensions: []string{"seq_q", "seq_kv"},
			CreateInputs: func(ctx ml.Context, be ml.Backend, _ string, _ []int64) []ml.Tensor {
				q := randomTensor(ctx, ml.DTypeF32, headDim, qH, seqQ, 1)
				kBytes := materializeTensor(be, ml.DTypeF16, headDim, kvH, seqKV, 1)
				vBytes := materializeTensor(be, ml.DTypeF16, headDim, kvH, seqKV, 1)
				k := ctx.Input().FromBytes(ml.DTypeF16, kBytes, headDim, kvH, seqKV, 1)
				v := ctx.Input().FromBytes(ml.DTypeF16, vBytes, headDim, kvH, seqKV, 1)
				return []ml.Tensor{q, k, v}
			},
			Run: func(ctx ml.Context, in []ml.Tensor) ml.Tensor {
				sdpa, ok := in[0].(ml.ScaledDotProductAttention)
				if !ok {
					return nil
				}
				scale := 1.0 / math.Sqrt(float64(headDim))
				// cacheConfigApplied=false so ScaledDotProductAttention will apply
				// CacheConfig transformations (mask dtype, etc.)
				return sdpa.ScaledDotProductAttention(ctx, in[1], in[2], nil, nil, nil, scale, false)
			},
		}

		pt := measureOpForBackend(backend, caps, opName, []int64{int64(seqQ), int64(seqKV)}, "f16", cfg)
		fmt.Printf("  %-40s → %10.1f us (stddev: %.1f, reps: %d)\n",
			exp.name, pt.LatencyUs, pt.StddevUs, pt.Reps)

		delete(opRegistry, opName)
	}

	fmt.Println()
	fmt.Println("Reference: actual inference per-layer FLASH_ATTN = ~58,356 us")

	return nil
}

func makeCausalMaskF32(ctx ml.Context, seqKV, seqQ int) ml.Tensor {
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
