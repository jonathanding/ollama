package perf

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// IsGGUFPath returns true if the ref looks like a file path rather than a model ID.
func IsGGUFPath(ref string) bool {
	return strings.HasSuffix(ref, ".gguf") ||
		strings.Contains(ref, "/") ||
		strings.Contains(ref, "\\")
}

// ResolveModelPath resolves a model reference to a local GGUF file path.
// If ref is a file path (ends with .gguf, or contains / or \), it's used directly.
// Otherwise, it's treated as an Ollama model ID and resolved via the local manifest.
func ResolveModelPath(ref string) (string, error) {
	if IsGGUFPath(ref) {
		if _, err := os.Stat(ref); err != nil {
			return "", fmt.Errorf("GGUF file not found: %s", ref)
		}
		return ref, nil
	}

	// Parse as Ollama model ID
	name := model.ParseName(ref)
	if !name.IsValid() {
		return "", fmt.Errorf("invalid model reference: %s", ref)
	}

	// ParseNamedManifest requires fully qualified name
	if !name.IsFullyQualified() {
		return "", fmt.Errorf("model name %q is not fully qualified", ref)
	}

	m, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return "", fmt.Errorf("model %q not found locally: %w (run 'ollama pull %s' first)", ref, err, ref)
	}

	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			blobPath, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return "", fmt.Errorf("cannot resolve blob path: %w", err)
			}
			if _, err := os.Stat(blobPath); err != nil {
				return "", fmt.Errorf("model blob missing: %s", blobPath)
			}
			return blobPath, nil
		}
	}

	return "", fmt.Errorf("no GGUF model layer found in manifest for %q", ref)
}
