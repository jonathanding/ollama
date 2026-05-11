package daop

import (
	"fmt"
	"log/slog"
)

// Probe extracts prompt embeddings for the MF scorer via hidden state extraction.
type Probe struct {
	hsProbe *HiddenStateProbe
}

func NewProbe(cfg *Config) (*Probe, error) {
	hsProbe, err := NewHiddenStateProbe(cfg.ProbeModel, cfg.ProbeLayer)
	if err != nil {
		return nil, fmt.Errorf("hidden state probe: %w", err)
	}

	slog.Info("daop: probe initialized",
		"model", cfg.ProbeModel, "layer", cfg.ProbeLayer)

	return &Probe{hsProbe: hsProbe}, nil
}

func (p *Probe) Extract(promptText string) ([]float32, error) {
	return p.hsProbe.Extract(promptText)
}

func (p *Probe) Close() {
	if p.hsProbe != nil {
		p.hsProbe.Close()
	}
}
