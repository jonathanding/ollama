package daop

import (
	"log/slog"
	"sync"
	"time"
)

// ProbeFunc extracts a prompt embedding from text. Injected dependency.
type ProbeFunc func(text string) ([]float32, error)

// Router orchestrates the DAOP routing decision.
type Router struct {
	cfg        *Config
	gate       *SubtaskGate
	classifier *SubtaskClassifier
	scorer     *MFScorer
	probe      ProbeFunc
	probeMu    sync.Mutex
}

func NewRouter(cfg *Config, gate *SubtaskGate, classifier *SubtaskClassifier, scorer *MFScorer, probe ProbeFunc) *Router {
	return &Router{
		cfg:        cfg,
		gate:       gate,
		classifier: classifier,
		scorer:     scorer,
		probe:      probe,
	}
}

func (r *Router) estimatePrefill(promptText string, result *DaopResult) {
	if r.cfg.PrefillMsPerByte > 0 {
		latEst := r.cfg.PrefillBaseMs + r.cfg.PrefillMsPerByte*float64(len(promptText))
		result.LatencyEstimate = &latEst
	}
}

// Route makes the offload/fallback decision for a chat request.
func (r *Router) Route(model string, promptText string, ctx *DaopContext) *DaopResult {
	t0 := time.Now()
	result := &DaopResult{
		Model:         model,
		Threshold:     r.cfg.AccuracyThreshold,
		GateThreshold: r.cfg.GateThreshold,
	}

	// Check if model is supported
	if !r.cfg.IsModelSupported(model) {
		return nil // nil means DAOP doesn't apply
	}

	// Step 1: Probe (needed for both classifier and MF scoring)
	r.probeMu.Lock()
	embedding, err := r.probe(promptText)
	r.probeMu.Unlock()

	if err != nil {
		slog.Warn("daop: probe failed, defaulting to offload", "error", err)
		result.Decision = "offload"
		result.RoutingMs = float64(time.Since(t0).Microseconds()) / 1000.0
		r.estimatePrefill(promptText, result)
		return result
	}

	// Step 2: Subtask gate
	subtask := ""
	if ctx != nil {
		subtask = ctx.Subtask
	}
	if subtask == "" && r.classifier != nil {
		predicted, conf := r.classifier.Predict(embedding)
		if predicted != "" {
			subtask = predicted
			confVal := conf
			result.ClassifierConf = &confVal
			slog.Debug("daop: classifier predicted subtask", "subtask", subtask, "confidence", conf)
		}
	}
	result.Subtask = subtask

	if subtask != "" {
		pass, rate := r.gate.Check(model, subtask)
		if rate >= 0 {
			result.GatePassRate = &rate
		}
		if !pass {
			result.Decision = "fallback"
			result.FallbackReason = "gate"
			result.RoutingMs = float64(time.Since(t0).Microseconds()) / 1000.0
			r.estimatePrefill(promptText, result)
			slog.Debug("daop: gate blocked", "model", model, "subtask", subtask, "rate", rate)
			return result
		}
	}

	// Step 3: MF scoring
	score, err := r.scorer.Score(model, embedding)
	if err != nil {
		slog.Warn("daop: scorer failed, defaulting to offload", "error", err)
		result.Decision = "offload"
		result.RoutingMs = float64(time.Since(t0).Microseconds()) / 1000.0
		r.estimatePrefill(promptText, result)
		return result
	}
	result.Confidence = &score

	// Step 4: Threshold check
	if score < r.cfg.AccuracyThreshold {
		result.Decision = "fallback"
		result.FallbackReason = "threshold"
		result.RoutingMs = float64(time.Since(t0).Microseconds()) / 1000.0
		r.estimatePrefill(promptText, result)
		slog.Debug("daop: below threshold", "model", model, "score", score)
		return result
	}

	result.Decision = "offload"
	result.RoutingMs = float64(time.Since(t0).Microseconds()) / 1000.0
	r.estimatePrefill(promptText, result)
	slog.Debug("daop: offload", "model", model, "score", score)
	return result
}
