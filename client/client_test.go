package client

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// shortTempDir returns a per-test temp dir under /tmp. macOS's $TMPDIR and
// Go's t.TempDir() produce paths that can exceed the 104-byte sockaddr_un
// limit, causing bind EINVAL for unrelated reasons. /tmp keeps paths short.
// Mirrors the helper in internal/ipc/uds_test.go; duplicated here to avoid
// adding a testutil package for a single reuse (see Plan A Task 12 notes).
func shortTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("/tmp", "runed-client-")
	if err != nil {
		t.Fatalf("mkdtemp: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

// TestConnect_EmbedReturnsVector is an integration test: requires the runed
// daemon to be running at ~/.runed/embedding.sock. Skipped unless
// RUNED_INTEGRATION=1 is set.
func TestConnect_EmbedReturnsVector(t *testing.T) {
	if os.Getenv("RUNED_INTEGRATION") == "" {
		t.Skip("set RUNED_INTEGRATION=1 and ensure daemon is running")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	c, err := Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer c.Close()

	vec, err := c.Embed(ctx, "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 1024 {
		t.Fatalf("want 1024, got %d", len(vec))
	}
}

// TestConnect_RejectsMissingDaemon verifies Connect fails clearly when no
// socket exists. Runs without RUNED_INTEGRATION.
//
// Uses WithNoSpawn() because Connect now auto-spawns on dial failure by
// default; without the opt-out this test would exercise the spawn path
// (which would fail looking up config, but for unrelated reasons).
func TestConnect_RejectsMissingDaemon(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	dir := shortTempDir(t)
	nowhere := filepath.Join(dir, "nonexistent.sock")

	c, err := Connect(ctx, WithSocketPath(nowhere), WithNoSpawn())
	if err == nil {
		c.Close()
		t.Fatal("expected error connecting to nonexistent socket, got nil")
	}
}

// TestConnect_WithNoSpawn_FailsFastWhenNoDaemon verifies that the explicit
// opt-out keeps Connect from trying to spawn — it surfaces the underlying
// dial/health error and returns within the caller's deadline.
func TestConnect_WithNoSpawn_FailsFastWhenNoDaemon(t *testing.T) {
	sockPath := filepath.Join(t.TempDir(), "embedding.sock")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := Connect(ctx, WithSocketPath(sockPath), WithNoSpawn())
	if err == nil {
		t.Fatal("expected error with WithNoSpawn and no daemon, got nil")
	}
}
