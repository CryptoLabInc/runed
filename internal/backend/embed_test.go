package backend

import (
	"context"
	"os"
	"testing"
	"time"
)

// Integration test: requires llama-server binary and GGUF via env vars (same as llama_test.go).
//
//	RUNED_TEST_LLAMA_SERVER=/path/to/llama-server
//	RUNED_TEST_GGUF=/path/to/qwen3-embedding-0.6b.f16.gguf
func TestLlamaBackend_EmbedReturns1024DimVector(t *testing.T) {
	srv := os.Getenv("RUNED_TEST_LLAMA_SERVER")
	gguf := os.Getenv("RUNED_TEST_GGUF")
	if srv == "" || gguf == "" {
		t.Skip("env not set")
	}

	b := NewLlamaBackend(Config{
		BinaryPath: srv,
		ModelPath:  gguf,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := b.Start(ctx); err != nil {
		t.Fatalf("start: %v", err)
	}
	defer b.Stop(context.Background())

	vec, err := b.Embed(ctx, "hello world", true)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(vec) != 1024 {
		t.Fatalf("want dim 1024, got %d", len(vec))
	}
}
