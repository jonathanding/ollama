package perf

import "math"

func elemSize(dtype string) float64 {
	switch dtype {
	case "f32":
		return 4
	case "f16":
		return 2
	case "bf16":
		return 2
	case "i8", "int8":
		return 1
	case "i32", "int32":
		return 4
	case "q4_0":
		return 0.5625
	case "q4_1":
		return 0.625
	case "q5_0":
		return 0.6875
	case "q5_1":
		return 0.75
	case "q8_0":
		return 1.0625
	case "q4_K":
		return 0.5625
	case "q5_K":
		return 0.6875
	case "q6_K":
		return 0.8125
	case "q3_K":
		return 0.4375
	case "iq4_nl":
		return 0.5625
	default:
		return 4
	}
}

func product(shape []int64) float64 {
	p := float64(1)
	for _, v := range shape {
		p *= float64(v)
	}
	return p
}

// ComputeFLOPs returns the estimated FLOPs for an op given input shapes.
func ComputeFLOPs(op string, shapes [][]int64) float64 {
	switch op {
	case "MUL_MAT":
		if len(shapes) < 2 || len(shapes[0]) < 2 || len(shapes[1]) < 2 {
			return 0
		}
		M := float64(shapes[0][0])
		K := float64(shapes[0][1])
		N := float64(shapes[1][1])
		batch := float64(1)
		if len(shapes[1]) > 2 {
			for i := 2; i < len(shapes[1]); i++ {
				batch *= float64(shapes[1][i])
			}
		}
		return 2 * M * K * N * batch
	case "MUL_MAT_ID":
		if len(shapes) < 2 {
			return 0
		}
		base := ComputeFLOPs("MUL_MAT", shapes[:2])
		experts := float64(1)
		if len(shapes) > 2 && len(shapes[2]) > 0 {
			experts = float64(shapes[2][0])
		}
		return base * experts
	case "FLASH_ATTN_EXT":
		if len(shapes) < 2 || len(shapes[0]) < 4 || len(shapes[1]) < 4 {
			return 0
		}
		B := float64(shapes[0][0])
		H := float64(shapes[0][1])
		Sq := float64(shapes[0][2])
		D := float64(shapes[0][3])
		Skv := float64(shapes[1][2])
		return 2 * B * H * Sq * Skv * D
	case "RMS_NORM":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0])
	case "LAYER_NORM":
		if len(shapes) < 1 {
			return 0
		}
		return 5 * product(shapes[0])
	case "SOFTMAX":
		if len(shapes) < 1 {
			return 0
		}
		return 4 * product(shapes[0])
	case "ADD", "MUL", "DIV", "NEG":
		if len(shapes) < 1 {
			return 0
		}
		return product(shapes[0])
	case "SILU", "GELU", "SIGMOID", "TANH":
		if len(shapes) < 1 {
			return 0
		}
		return 5 * product(shapes[0])
	case "GLU", "SWIGLU", "GEGLU":
		if len(shapes) < 1 {
			return 0
		}
		return 6 * product(shapes[0])
	case "ROPE":
		if len(shapes) < 1 || len(shapes[0]) < 4 {
			return 0
		}
		B := float64(shapes[0][0])
		H := float64(shapes[0][1])
		S := float64(shapes[0][2])
		D := float64(shapes[0][3])
		return 6 * B * H * S * (D / 2)
	case "GET_ROWS":
		return 0
	case "CONT", "CPY", "CONCAT":
		return 0
	case "CONV_2D":
		if len(shapes) < 1 || len(shapes[0]) < 7 {
			return 0
		}
		p := float64(2)
		for _, v := range shapes[0] {
			p *= float64(v)
		}
		return p
	case "EXP", "SQRT", "SQR", "SIN", "COS":
		if len(shapes) < 1 {
			return 0
		}
		return product(shapes[0])
	case "SUM_ROWS":
		if len(shapes) < 1 || len(shapes[0]) < 2 {
			return 0
		}
		return float64(shapes[0][0]) * float64(shapes[0][1])
	case "TOP_K":
		if len(shapes) < 1 || len(shapes[0]) < 2 {
			return 0
		}
		N := float64(shapes[0][0])
		K := float64(shapes[0][1])
		return N * math.Log2(K)
	case "L2_NORM":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0])
	case "SOFTPLUS":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0])
	case "CUM_SUM":
		if len(shapes) < 1 {
			return 0
		}
		return product(shapes[0])
	case "VIEW", "RESHAPE", "PERMUTE":
		return 0
	default:
		return 0
	}
}

// ComputeBytes returns the estimated bytes moved for an op.
func ComputeBytes(op string, shapes [][]int64, computeDtype, weightDtype string) float64 {
	es := elemSize(computeDtype)
	ws := elemSize(weightDtype)
	if weightDtype == "" {
		ws = es
	}

	switch op {
	case "MUL_MAT":
		if len(shapes) < 2 || len(shapes[0]) < 2 || len(shapes[1]) < 2 {
			return 0
		}
		M := float64(shapes[0][0])
		K := float64(shapes[0][1])
		N := float64(shapes[1][1])
		batch := float64(1)
		if len(shapes[1]) > 2 {
			for i := 2; i < len(shapes[1]); i++ {
				batch *= float64(shapes[1][i])
			}
		}
		return (ws*M*K + es*K*N + 4*M*N) * batch
	case "MUL_MAT_ID":
		if len(shapes) < 2 {
			return 0
		}
		base := ComputeBytes("MUL_MAT", shapes[:2], computeDtype, weightDtype)
		experts := float64(1)
		if len(shapes) > 2 && len(shapes[2]) > 0 {
			experts = float64(shapes[2][0])
		}
		return base * experts
	case "FLASH_ATTN_EXT":
		if len(shapes) < 2 || len(shapes[0]) < 4 || len(shapes[1]) < 4 {
			return 0
		}
		B := float64(shapes[0][0])
		H := float64(shapes[0][1])
		Sq := float64(shapes[0][2])
		D := float64(shapes[0][3])
		Skv := float64(shapes[1][2])
		return B * H * (Sq*D + 2*Skv*D + Sq*D) * es
	case "RMS_NORM":
		if len(shapes) < 1 {
			return 0
		}
		total := product(shapes[0])
		N := float64(shapes[0][0])
		return 2*total*es + N*es
	case "LAYER_NORM":
		if len(shapes) < 1 {
			return 0
		}
		total := product(shapes[0])
		N := float64(shapes[0][0])
		return 2*total*es + N*es + N*es
	case "SOFTMAX":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	case "ADD", "MUL", "DIV":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0]) * es
	case "NEG":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	case "SILU", "GELU", "SIGMOID", "TANH":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	case "GLU", "SWIGLU", "GEGLU":
		if len(shapes) < 1 {
			return 0
		}
		return 3 * product(shapes[0]) * es
	case "GET_ROWS":
		if len(shapes) < 1 || len(shapes[0]) < 2 {
			return 0
		}
		Nidx := float64(shapes[0][0])
		D := float64(shapes[0][1])
		return Nidx*D*ws + Nidx*D*4
	case "ROPE":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	case "CONT", "CPY", "CONCAT":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	case "CONV_2D":
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	case "VIEW", "RESHAPE", "PERMUTE":
		return 0
	default:
		if len(shapes) < 1 {
			return 0
		}
		return 2 * product(shapes[0]) * es
	}
}

func IsZeroCostOp(op string) bool {
	switch op {
	case "VIEW", "RESHAPE", "PERMUTE":
		return true
	default:
		return false
	}
}

func CanComputeFLOPs(op string) bool {
	switch op {
	case "MUL_MAT", "MUL_MAT_ID", "FLASH_ATTN_EXT",
		"RMS_NORM", "LAYER_NORM", "SOFTMAX",
		"ADD", "MUL", "DIV", "NEG",
		"SILU", "GELU", "SIGMOID", "TANH",
		"GLU", "SWIGLU", "GEGLU",
		"ROPE", "CONV_2D",
		"EXP", "SQRT", "SQR", "SIN", "COS",
		"SUM_ROWS", "TOP_K", "L2_NORM", "SOFTPLUS", "CUM_SUM":
		return true
	default:
		return false
	}
}
