package perf

// mapWeightDtype maps unsupported K-quant and other weight dtypes to the nearest
// measured dtype. The Go DType abstraction only exposes f32/f16/q4_0/q8_0, but real
// models use q4_K, q5_K, q6_K etc.
func mapWeightDtype(wdt string) string {
	switch wdt {
	case "f32", "f16", "q4_0", "q8_0":
		return wdt
	case "q4_K", "q4_1":
		return "q4_0"
	case "q5_K", "q5_0", "q5_1", "q6_K":
		return "q8_0"
	case "q3_K", "q2_K":
		return "q4_0"
	case "q8_K":
		return "q8_0"
	default:
		return "f16"
	}
}
