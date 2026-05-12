package daop

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

type CloudConfig struct {
	BaseURL string
	APIKey  string
	Model   string
}

func LoadCloudConfig() *CloudConfig {
	key := os.Getenv("LLM_API_KEY_ALI")
	base := os.Getenv("LLM_BASE_URL_ALI")
	model := os.Getenv("LLM_MODEL_ALI")
	if key == "" || base == "" {
		return nil
	}
	if model == "" {
		model = "qwen-max"
	}
	return &CloudConfig{BaseURL: base, APIKey: key, Model: model}
}

type CloudMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type cloudRequest struct {
	Model    string         `json:"model"`
	Messages []CloudMessage `json:"messages"`
	Stream   bool           `json:"stream"`
}

type cloudChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
}

type cloudChunk struct {
	Choices []cloudChoice `json:"choices"`
}

// StreamCloud calls the cloud LLM and sends tokens via callback.
// Returns the full response text accumulated.
func StreamCloud(cfg *CloudConfig, messages []CloudMessage, onToken func(content string)) (string, error) {
	body, err := json.Marshal(cloudRequest{
		Model:    cfg.Model,
		Messages: messages,
		Stream:   true,
	})
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", cfg.BaseURL+"/chat/completions", strings.NewReader(string(body)))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("cloud request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		errBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("cloud API returned %d: %s", resp.StatusCode, string(errBody)[:min(len(errBody), 200)])
	}

	slog.Info("daop: cloud streaming started", "model", cfg.Model)

	scanner := bufio.NewScanner(resp.Body)
	var fullText strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		if strings.TrimSpace(data) == "[DONE]" {
			break
		}

		var chunk cloudChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		content := chunk.Choices[0].Delta.Content
		if content != "" {
			fullText.WriteString(content)
			onToken(content)
		}
	}

	return fullText.String(), scanner.Err()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
