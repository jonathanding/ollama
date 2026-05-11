package daop

import (
	"os"
	"testing"
)

func BenchmarkHiddenStateProbe_Extract_Short(b *testing.B) {
	if _, err := os.Stat(testGGUFPath); err != nil {
		b.Skipf("GGUF file not available: %v", err)
	}

	probe, err := NewHiddenStateProbe(testGGUFPath, 14)
	if err != nil {
		b.Fatalf("NewHiddenStateProbe: %v", err)
	}
	defer probe.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := probe.Extract("What is 2+2?")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHiddenStateProbe_Extract_Long(b *testing.B) {
	if _, err := os.Stat(testGGUFPath); err != nil {
		b.Skipf("GGUF file not available: %v", err)
	}

	probe, err := NewHiddenStateProbe(testGGUFPath, 14)
	if err != nil {
		b.Fatalf("NewHiddenStateProbe: %v", err)
	}
	defer probe.Close()

	longPrompt := "Write a detailed explanation of how quantum entanglement works, including the EPR paradox, Bell's theorem, and modern applications in quantum computing and quantum cryptography. Please provide specific examples and mathematical formulations where appropriate."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := probe.Extract(longPrompt)
		if err != nil {
			b.Fatal(err)
		}
	}
}
