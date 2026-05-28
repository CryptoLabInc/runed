//go:build !windows

package backend

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os/exec"
	"strconv"
	"testing"
	"time"
)

// TestStopWaitsForInflightEmbed verifies that Stop blocks until in-flight
// Embed/EmbedBatch HTTP calls finish — the race fix's core invariant. Without
// inflightMu, Stop would SIGTERM llama-server while a request was mid-flight
// and the client would see a connection reset.
//
// Setup uses a fake llama-server (httptest.Server that blocks on a channel)
// and a real `sleep` child as the SIGTERM target. Real llama-server isn't
// needed: the lock interaction is the entire subject under test.
func TestStopWaitsForInflightEmbed(t *testing.T) {
	received := make(chan struct{})
	release := make(chan struct{})
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(received)
		<-release
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3]}]}`))
	}))
	defer fake.Close()

	host, port := splitHostPort(t, fake.URL)
	b := NewLlamaBackend(Config{Host: host})

	cmd := startPlaceholderChild(t)
	defer cleanupPlaceholderChild(cmd)
	b.mu.Lock()
	b.port = port
	b.cmd = cmd
	b.mu.Unlock()

	embedDone := make(chan error, 1)
	go func() {
		_, err := b.Embed(context.Background(), "hello", true)
		embedDone <- err
	}()

	select {
	case <-received:
	case <-time.After(2 * time.Second):
		t.Fatal("Embed never reached fake llama-server")
	}

	stopDone := make(chan error, 1)
	go func() {
		stopDone <- b.Stop(context.Background())
	}()

	// Stop must block until the in-flight Embed finishes. 200ms is generous —
	// without the writer Lock it returns immediately (SIGTERM is non-blocking).
	select {
	case err := <-stopDone:
		t.Fatalf("Stop returned before Embed finished (race not fixed): err=%v", err)
	case <-time.After(200 * time.Millisecond):
	}

	close(release)

	select {
	case err := <-embedDone:
		if err != nil {
			t.Fatalf("Embed failed instead of completing under RLock: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Embed never completed after release")
	}

	select {
	case err := <-stopDone:
		if err != nil {
			t.Fatalf("Stop returned error: %v", err)
		}
	case <-time.After(6 * time.Second):
		t.Fatal("Stop never completed after Embed released")
	}
}

// TestEmbedReturnsErrNotStartedAfterStop is the contract server.go relies on
// for its retry loop: once Stop has run, Embed must return ErrNotStarted
// (not some other error) so the server can recognize the race-recoverable case.
func TestEmbedReturnsErrNotStartedAfterStop(t *testing.T) {
	b := NewLlamaBackend(Config{})

	cmd := startPlaceholderChild(t)
	defer cleanupPlaceholderChild(cmd)
	b.mu.Lock()
	b.port = 1
	b.cmd = cmd
	b.mu.Unlock()

	if err := b.Stop(context.Background()); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	_, err := b.Embed(context.Background(), "hello", true)
	if !errors.Is(err, ErrNotStarted) {
		t.Fatalf("want ErrNotStarted, got %v", err)
	}

	_, err = b.EmbedBatch(context.Background(), []string{"hello"}, true)
	if !errors.Is(err, ErrNotStarted) {
		t.Fatalf("EmbedBatch: want ErrNotStarted, got %v", err)
	}
}

func splitHostPort(t *testing.T, raw string) (string, int) {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	host, portStr, err := net.SplitHostPort(u.Host)
	if err != nil {
		t.Fatalf("split host:port: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("port: %v", err)
	}
	return host, port
}

// startPlaceholderChild spawns a long-running `sleep` so Stop has a real
// Process to SIGTERM — exercises the lock path without needing llama-server.
func startPlaceholderChild(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.Command("sleep", "60")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn placeholder: %v", err)
	}
	return cmd
}

func cleanupPlaceholderChild(cmd *exec.Cmd) {
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}
}
