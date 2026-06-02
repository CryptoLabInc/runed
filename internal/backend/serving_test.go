//go:build !windows

package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"testing"
	"time"
)

// TestServing_DistinguishesIdleFromDegraded checks the Serving mapping given a
// known child state: an intentional idle-suspend is ServingIdle (not a fault)
// while a crash or an up-but-unhealthy child is ServingDegraded. This is what
// lets Health surface STATUS_IDLE instead of misleading users with
// STATUS_DEGRADED after the idle ticker stops llama-server to free memory.
func TestServing_DistinguishesIdleFromDegraded(t *testing.T) {
	t.Run("child down after intentional suspend → idle", func(t *testing.T) {
		b := NewLlamaBackend(Config{})
		b.mu.Lock()
		b.cmd = nil
		b.state = childSuspended
		b.mu.Unlock()
		if got := b.Serving(context.Background()); got != ServingIdle {
			t.Fatalf("Serving() = %v, want ServingIdle", got)
		}
	})

	t.Run("child down after unexpected exit → degraded", func(t *testing.T) {
		b := NewLlamaBackend(Config{})
		b.mu.Lock()
		b.cmd = nil
		b.state = childFailed
		b.mu.Unlock()
		if got := b.Serving(context.Background()); got != ServingDegraded {
			t.Fatalf("Serving() = %v, want ServingDegraded", got)
		}
	})

	t.Run("child up and healthy → ok", func(t *testing.T) {
		fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer fake.Close()
		host, port := splitHostPort(t, fake.URL)
		b := NewLlamaBackend(Config{Host: host})
		b.mu.Lock()
		b.cmd = &exec.Cmd{} // non-nil → child considered up
		b.port = port
		b.state = childRunning
		b.mu.Unlock()
		if got := b.Serving(context.Background()); got != ServingOK {
			t.Fatalf("Serving() = %v, want ServingOK", got)
		}
	})

	t.Run("child up but unhealthy → degraded", func(t *testing.T) {
		fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer fake.Close()
		host, port := splitHostPort(t, fake.URL)
		b := NewLlamaBackend(Config{Host: host})
		b.mu.Lock()
		b.cmd = &exec.Cmd{} // non-nil → child considered up
		b.port = port
		b.state = childRunning
		b.mu.Unlock()
		if got := b.Serving(context.Background()); got != ServingDegraded {
			t.Fatalf("Serving() = %v, want ServingDegraded", got)
		}
	})
}

// TestServing_LifecycleTransitions drives the real child lifecycle (via a
// placeholder `sleep` process) to verify which teardown paths produce IDLE vs
// DEGRADED. The key regression: only Stop() (the idle ticker) yields IDLE — an
// internal stopLocked from a failed (re)start, or a crash, must stay DEGRADED.
func TestServing_LifecycleTransitions(t *testing.T) {
	t.Run("Stop (idle-suspend) → idle", func(t *testing.T) {
		b := NewLlamaBackend(Config{})
		cmd := attachPlaceholderChild(t, b, 1)
		defer cleanupPlaceholderChild(cmd)
		if err := b.Stop(context.Background()); err != nil {
			t.Fatalf("Stop: %v", err)
		}
		if got := b.Serving(context.Background()); got != ServingIdle {
			t.Fatalf("after Stop: got %v, want ServingIdle", got)
		}
	})

	t.Run("internal stopLocked (failed start/restart) → degraded, not idle", func(t *testing.T) {
		b := NewLlamaBackend(Config{})
		cmd := attachPlaceholderChild(t, b, 1)
		defer cleanupPlaceholderChild(cmd)
		// startLocked / EnsureStarted stop the child via stopLocked (not Stop)
		// when a (re)start fails. That must NOT be reported as an idle-suspend.
		b.lifecycleMu.Lock()
		_ = b.stopLocked(context.Background())
		b.lifecycleMu.Unlock()
		if got := b.Serving(context.Background()); got != ServingDegraded {
			t.Fatalf("internal stop should be DEGRADED, got %v", got)
		}
	})

	t.Run("unexpected exit (crash) → degraded", func(t *testing.T) {
		b := NewLlamaBackend(Config{})
		cmd := attachPlaceholderChild(t, b, 1)
		defer cleanupPlaceholderChild(cmd)
		_ = cmd.Process.Kill() // crash: no Stop/stopLocked involved
		deadline := time.Now().Add(2 * time.Second)
		for b.getCmd() != nil && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if got := b.Serving(context.Background()); got != ServingDegraded {
			t.Fatalf("after crash: got %v, want ServingDegraded", got)
		}
	})
}
