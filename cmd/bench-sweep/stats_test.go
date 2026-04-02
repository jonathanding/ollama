package main

import (
	"testing"
)

func TestComputeStats_OddSlice(t *testing.T) {
	s := computeStats([]float64{100, 200, 300, 400, 500})
	if s.Mean != 300 {
		t.Errorf("mean: got %.2f, want 300", s.Mean)
	}
	if s.Median != 300 {
		t.Errorf("median: got %.2f, want 300", s.Median)
	}
	// p99: ceil(5*0.99)-1 = 5-1 = 4, sorted[4] = 500
	if s.P99 != 500 {
		t.Errorf("p99: got %.2f, want 500", s.P99)
	}
}

func TestComputeStats_EvenSlice(t *testing.T) {
	// median of [1,3,5,7] = (3+5)/2 = 4
	s := computeStats([]float64{1, 3, 5, 7})
	if s.Median != 4 {
		t.Errorf("median: got %.2f, want 4", s.Median)
	}
}

func TestComputeStats_ConstantCV(t *testing.T) {
	s := computeStats([]float64{100, 100, 100, 100})
	if s.CVPct != 0 {
		t.Errorf("CV for constant slice: got %.4f, want 0", s.CVPct)
	}
	if s.Stddev != 0 {
		t.Errorf("stddev for constant slice: got %.4f, want 0", s.Stddev)
	}
}

func TestComputeStats_Empty(t *testing.T) {
	s := computeStats(nil)
	// Should not panic; zero values acceptable
	_ = s
}

func TestComputeStats_Single(t *testing.T) {
	s := computeStats([]float64{42})
	if s.Mean != 42 {
		t.Errorf("mean: got %.2f, want 42", s.Mean)
	}
	if s.CVPct != 0 {
		t.Errorf("CV for single value: got %.4f, want 0", s.CVPct)
	}
}

func TestComputeStats_KnownCV(t *testing.T) {
	// mean=100, stddev≈50, CV≈50%
	s := computeStats([]float64{50, 100, 150})
	if s.Mean != 100 {
		t.Errorf("mean: got %.2f, want 100", s.Mean)
	}
	if s.CVPct < 40 || s.CVPct > 60 {
		t.Errorf("CV: got %.2f, want ~50", s.CVPct)
	}
}
