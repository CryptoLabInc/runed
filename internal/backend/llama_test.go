package backend

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Integration test: requires env vars pointing at real llama-server binary and GGUF.
//
//	RUNED_TEST_LLAMA_SERVER=/path/to/llama-server
//	RUNED_TEST_GGUF=/path/to/qwen3-embedding-0.6b.f16.gguf
func TestLlamaBackend_SpawnAndHealthy(t *testing.T) {
	srv := os.Getenv("RUNED_TEST_LLAMA_SERVER")
	gguf := os.Getenv("RUNED_TEST_GGUF")
	if srv == "" || gguf == "" {
		t.Skip("RUNED_TEST_LLAMA_SERVER / RUNED_TEST_GGUF not set")
	}
	if _, err := os.Stat(srv); err != nil {
		t.Fatalf("llama-server binary missing: %v", err)
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Fatalf("gguf missing: %v", err)
	}

	logFile := filepath.Join(t.TempDir(), "llama.log")
	b := NewLlamaBackend(Config{
		BinaryPath: srv,
		ModelPath:  gguf,
		LogPath:    logFile,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := b.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer b.Stop(context.Background())

	if b.Port() == 0 {
		t.Fatal("port not captured")
	}
	if !b.IsHealthy(ctx) {
		t.Fatal("health check failed")
	}
}

// TestLlamaBackend_StopIsIdempotent exercises pure logic: Stop before Start
// must be a no-op, and repeated Stop calls must remain no-ops.
func TestLlamaBackend_StopIsIdempotent(t *testing.T) {
	b := NewLlamaBackend(Config{})
	// Before Start: Stop is a no-op.
	if err := b.Stop(context.Background()); err != nil {
		t.Fatalf("Stop before Start: %v", err)
	}
	// Double-Stop after never-started: still no-op.
	if err := b.Stop(context.Background()); err != nil {
		t.Fatalf("double-Stop before Start: %v", err)
	}
}

// TestLlamaBackend_StartFailsOnBadModel ensures Start surfaces an error
// promptly when llama-server exits immediately (e.g. invalid GGUF) instead
// of hanging until context deadline.
func TestLlamaBackend_StartFailsOnBadModel(t *testing.T) {
	srv := os.Getenv("RUNED_TEST_LLAMA_SERVER")
	if srv == "" {
		t.Skip("RUNED_TEST_LLAMA_SERVER not set")
	}
	bad := filepath.Join(t.TempDir(), "not-a-model.gguf")
	if err := os.WriteFile(bad, []byte("not a gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	b := NewLlamaBackend(Config{BinaryPath: srv, ModelPath: bad})
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err := b.Start(ctx)
	if err == nil {
		b.Stop(context.Background())
		t.Fatal("expected Start to fail on bad model, got nil")
	}
	// Should fail quickly (well under 10s), NOT via ctx deadline
	if ctx.Err() != nil {
		t.Fatalf("Start hung until ctx deadline: %v", err)
	}
}
