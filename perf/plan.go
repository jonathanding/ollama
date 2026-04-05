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
//
// The ops list fully controls which benchmarks run: regular ops become StepOperator,
// fused ops become StepFusedOp, MUL_MAT becomes StepMulMatRef. Orchestration overhead
// runs only when the full default op set is used.
func buildBenchmarkPlan(ops []string, dtypes []string, caps BackendCapabilities, cfg BenchmarkConfig) BenchmarkPlan {
	var plan BenchmarkPlan

	// 1. Hardware characterization — first unless skipped
	if !cfg.SkipHWChar {
		plan = append(plan, BenchmarkStep{Type: StepHWChar})
	}

	// Fused ops are routed to StepFusedOp instead of StepOperator
	fusedOps := map[string]bool{
		"RMS_NORM_MUL":      true,
		"RMS_NORM_MUL_ROPE": true,
		"MUL_MAT_ADD":       true,
	}

	// Build a set for quick lookup
	opsSet := make(map[string]bool, len(ops))
	for _, op := range ops {
		opsSet[op] = true
	}

	// 2. All operator benchmarks — unified loop
	for _, op := range ops {
		// Fused ops: route to StepFusedOp (requires backend fusion support)
		if fusedOps[op] {
			if len(caps.FusionRules) == 0 {
				continue
			}
			if _, ok := LookupRegistry(op); !ok {
				continue
			}
			if op == "MUL_MAT_ADD" {
				for _, wdt := range Phase1Dtypes() {
					plan = append(plan, BenchmarkStep{
						Type:        StepFusedOp,
						Op:          op,
						Dtype:       "f32",
						WeightDtype: wdt,
						FixedDims:   map[string]int64{"M": 4096, "K": 4096},
					})
				}
			} else {
				plan = append(plan, BenchmarkStep{
					Type:  StepFusedOp,
					Op:    op,
					Dtype: "f32",
				})
			}
			continue
		}

		// MUL_MAT: generate reference curves per weight dtype × (M,K) grid
		if op == "MUL_MAT" {
			for _, wdt := range Phase1Dtypes() {
				for _, dims := range Phase1MulMatFixedDims() {
					plan = append(plan, BenchmarkStep{
						Type:        StepMulMatRef,
						Op:          "MUL_MAT",
						WeightDtype: wdt,
						FixedDims:   map[string]int64{"M": dims[0], "K": dims[1]},
					})
				}
			}
			continue
		}

		// Regular ops
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

	// 3. Orchestration overhead — only for full calibration (not custom --ops)
	if caps.HasGPUTimestamp && opsSet["ORCHESTRATION_OVERHEAD"] {
		plan = append(plan, BenchmarkStep{Type: StepOverhead})
	}

	return plan
}
