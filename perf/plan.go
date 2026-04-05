package perf

import "log/slog"

// StepType identifies the kind of benchmark step in a BenchmarkPlan.
type StepType int

const (
	StepHWChar    StepType = iota // Hardware characterization (peak TOPS, BW)
	StepMulMatRef                 // MUL_MAT reference curve for one weight dtype
	StepOperator                  // Single op benchmark (1D or multi-dim)
	StepFusedOp                   // Fused op benchmark (RMS_NORM_MUL, etc.)
	StepOverhead                  // Orchestration overhead benchmark
)

// BenchmarkStep is one unit of work in a BenchmarkPlan.
type BenchmarkStep struct {
	Type        StepType
	Op          string           // for StepOperator/StepFusedOp/StepMulMatRef
	Dtype       string           // compute dtype
	WeightDtype string           // for MUL_MAT: weight dtype
	FixedDims   map[string]int64 // for multi-dim ops
}

// BenchmarkPlan is an ordered list of steps that RunBenchmark executes uniformly.
type BenchmarkPlan []BenchmarkStep

// buildBenchmarkPlan creates the complete list of benchmark steps based on parameters.
// This is the single point where filtering and ordering decisions are made.
// RunBenchmark simply iterates this list — no scattered conditionals.
func buildBenchmarkPlan(ops []string, dtypes []string, caps BackendCapabilities) BenchmarkPlan {
	var plan BenchmarkPlan

	// 1. Hardware characterization — always first
	plan = append(plan, BenchmarkStep{Type: StepHWChar})

	// Known fused ops — benchmarked separately, not in the main op loop
	fusedOps := map[string]bool{
		"RMS_NORM_MUL":      true,
		"RMS_NORM_MUL_ROPE": true,
		"MUL_MAT_ADD":       true,
	}

	// 2. Main operator benchmarks
	for _, op := range ops {
		if fusedOps[op] {
			continue
		}

		if op == "MUL_MAT" {
			for _, wdt := range Phase1Dtypes() {
				plan = append(plan, BenchmarkStep{
					Type:        StepMulMatRef,
					Op:          "MUL_MAT",
					WeightDtype: wdt,
					FixedDims:   map[string]int64{"M": 4096, "K": 4096},
				})
			}
			continue
		}

		opDtypes := dtypes
		if op == "FLASH_ATTN_EXT" {
			opDtypes = []string{"f16"}
		}
		runner, ok := LookupRegistry(op)
		if !ok {
			slog.Warn("skipping unknown op in plan", "op", op)
			continue
		}
		if len(runner.Dimensions) == 1 && runner.Dimensions[0] == "N" {
			opDtypes = []string{"f32"}
		}

		for _, dtype := range opDtypes {
			grids := buildSamplingGrids(op, dtype, "")
			for _, grid := range grids {
				plan = append(plan, BenchmarkStep{
					Type:      StepOperator,
					Op:        op,
					Dtype:     dtype,
					FixedDims: grid.FixedDims,
				})
			}
		}
	}

	// 3. Fused op benchmarks — if backend supports fusion
	if len(caps.FusionRules) > 0 {
		fusedList := []string{"RMS_NORM_MUL", "RMS_NORM_MUL_ROPE", "MUL_MAT_ADD"}
		for _, fop := range fusedList {
			if _, ok := LookupRegistry(fop); !ok {
				continue
			}
			if fop == "MUL_MAT_ADD" {
				for _, wdt := range Phase1Dtypes() {
					plan = append(plan, BenchmarkStep{
						Type:        StepFusedOp,
						Op:          fop,
						Dtype:       "f32",
						WeightDtype: wdt,
						FixedDims:   map[string]int64{"M": 4096, "K": 4096},
					})
				}
			} else {
				plan = append(plan, BenchmarkStep{
					Type:  StepFusedOp,
					Op:    fop,
					Dtype: "f32",
				})
			}
		}
	}

	// 4. Orchestration overhead — for GPU backends with timestamp support
	if caps.HasGPUTimestamp {
		plan = append(plan, BenchmarkStep{Type: StepOverhead})
	}

	return plan
}
