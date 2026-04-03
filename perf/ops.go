package perf

// elemSize returns the bytes per element for a dtype string.
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

// product returns the product of all elements in a shape slice.
func product(shape []int64) float64 {
	p := float64(1)
	for _, v := range shape {
		p *= float64(v)
	}
	return p
}

// IsZeroCostOp returns true for ops that don't consume compute time
// (metadata-only operations like view, reshape, permute).
func IsZeroCostOp(op string) bool {
	switch op {
	case "VIEW", "RESHAPE", "PERMUTE":
		return true
	default:
		return false
	}
}
