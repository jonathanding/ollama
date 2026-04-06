package perf

// mapWeightDtype maps model weight dtypes to the nearest benchmark-measured dtype.
// K-quant types that are directly benchmarked pass through as identity.
func mapWeightDtype(wdt string) string {
	switch wdt {
	case "f32", "f16", "q4_0", "q8_0", "q4_K", "q6_K":
		return wdt
	case "q4_1":
		return "q4_0"
	case "q5_K", "q5_0", "q5_1":
		return "q6_K"
	case "q3_K", "q2_K":
		return "q4_0"
	case "q8_K":
		return "q8_0"
	default:
		return "f16"
	}
}

// dtypeFallback returns the approximate fallback dtype for when direct calibration
// curves are not yet available. Returns "" if no fallback exists (base dtype).
// Used by lookupLatencyV3 to provide estimates before re-benchmarking.
func dtypeFallback(wdt string) string {
	switch wdt {
	case "q4_K":
		return "q4_0"
	case "q6_K":
		return "q8_0"
	default:
		return ""
	}
}
