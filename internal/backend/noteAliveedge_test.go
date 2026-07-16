//go:build !windows

package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// TestQuickProbeStampEdge measures the "idle → slow first embed → concurrent
// second request" edge that the EnsureStarted quick-probe stamp covers. It
// times the SECOND request's EnsureStarted while the first embed holds the
// slot and /health is starved (quick probe misses). Current code stamps
// lastAlive on the first request's quick probe, so the second short-circuits
// fast; moving noteAlive to IsHealthy drops that stamp, so the second escalates
// into restartIfDead (extra verdict-probe latency). The test asserts the second
// request stays bounded either way (no false restart, self-correcting) and
// prints the latency so the behavior difference is visible.
func TestQuickProbeStampEdge(t *testing.T) {
	var inflight atomic.Bool
	embedStarted := make(chan struct{})
	embedRelease := make(chan struct{})
	var startedOnce atomic.Bool

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if inflight.Load() {
			// Starved under load: block past quickProbeTimeout (500ms) but
			// recover within verdictProbeTimeout (2s).
			select {
			case <-time.After(1200 * time.Millisecond):
			case <-r.Context().Done():
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		inflight.Store(true)
		if !startedOnce.Swap(true) {
			close(embedStarted)
		}
		<-embedRelease
		inflight.Store(false)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3]}]}`))
	})
	fake := httptest.NewServer(mux)
	defer fake.Close()
	defer close(embedRelease)

	host, port := splitHostPort(t, fake.URL)
	b := NewLlamaBackend(Config{Host: host, BinaryPath: "/usr/bin/false"})
	b.daemonCtx = context.Background()
	cmd := attachPlaceholderChild(t, b, port)
	defer cleanupPlaceholderChild(cmd)

	// Request A: EnsureStarted (idle server → quick probe succeeds) then Embed
	// (holds the slot). Mirrors server.go's Embed handler.
	go func() {
		if err := b.EnsureStarted(); err != nil {
			t.Errorf("A EnsureStarted: %v", err)
			return
		}
		_, _ = b.Embed(context.Background(), "A", true)
	}()

	select {
	case <-embedStarted:
	case <-time.After(3 * time.Second):
		t.Fatal("embed A never reached the server")
	}

	// Request B: time its EnsureStarted while A holds the slot and /health is starved.
	start := time.Now()
	err := b.EnsureStarted()
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("B EnsureStarted returned error (unexpected restart/kill?): %v", err)
	}
	if b.getCmd() != cmd {
		t.Fatal("B path restarted the child — false kill of a busy server")
	}
	t.Logf("B EnsureStarted latency while server busy: %v", elapsed.Round(time.Millisecond))
	if elapsed > 5*time.Second {
		t.Fatalf("B blocked too long (%v) — likely stuck on drain", elapsed)
	}
}
