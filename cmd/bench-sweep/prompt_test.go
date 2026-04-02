package main

import (
	"testing"
)

func TestPromptText_ExactLength(t *testing.T) {
	for _, n := range []int{100, 500, 1000, 4000} {
		got := promptText(n, 0)
		if len(got) != n {
			t.Errorf("promptText(%d, 0): got len %d, want %d", n, len(got), n)
		}
	}
}

func TestPromptText_VariesByEpoch(t *testing.T) {
	t0 := promptText(500, 0)
	t1 := promptText(500, 1)
	t2 := promptText(500, 2)
	if t0 == t1 || t1 == t2 || t0 == t2 {
		t.Error("expected different text for different epochs")
	}
}

func TestPromptText_DifferentPrefix(t *testing.T) {
	// The first 100 chars must differ across epochs to defeat KV cache matching
	t0 := promptText(500, 0)
	t1 := promptText(500, 1)
	if len(t0) >= 100 && len(t1) >= 100 && t0[:100] == t1[:100] {
		t.Error("epoch 0 and 1 share first 100 chars — KV cache defeat may fail")
	}
}

func TestPromptText_LargerThanCorpus(t *testing.T) {
	big := len(corpus)*2 + 100
	got := promptText(big, 0)
	if len(got) != big {
		t.Errorf("expected %d chars, got %d", big, len(got))
	}
}

func TestPromptText_Zero(t *testing.T) {
	if promptText(0, 0) != "" {
		t.Error("expected empty string for zero length")
	}
}

func TestCalibrateChars_ScalesUp(t *testing.T) {
	// asked 4000 chars → got 800 tokens, want 1000 → scale to 5000
	result := calibrateChars(4000, 1000, 800)
	if result != 5000 {
		t.Errorf("got %d, want 5000", result)
	}
}

func TestCalibrateChars_ScalesDown(t *testing.T) {
	// asked 4000 chars → got 1200 tokens, want 1000 → scale to 3333
	result := calibrateChars(4000, 1000, 1200)
	if result != 3333 {
		t.Errorf("got %d, want 3333", result)
	}
}

func TestCalibrateChars_ZeroActual(t *testing.T) {
	// Should return unchanged when actualTokens=0
	result := calibrateChars(4000, 1000, 0)
	if result != 4000 {
		t.Errorf("got %d, want 4000", result)
	}
}

func TestInitialChars(t *testing.T) {
	if initialChars(512) != 2048 {
		t.Errorf("expected 2048 for 512 tokens, got %d", initialChars(512))
	}
	if initialChars(4096) != 16384 {
		t.Errorf("expected 16384 for 4096 tokens, got %d", initialChars(4096))
	}
}
