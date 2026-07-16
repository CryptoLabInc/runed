//go:build !windows

package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// These tests pin the busy-vs-dead health verdict. The production failure
// they guard against: a saturated single-slot llama-server starves its
// /health endpoint, the old EnsureStarted treated one missed 500ms probe as
// death and killed the child WITHOUT draining in-flight embeds — severing
// them with EOF — and the restarted child saturated again immediately,
// looping every ~6s.

// newVerdictBackend wires a fake llama-server and a placeholder child into a
// backend whose daemonCtx is set, mirroring the state after a real Start.
func newVerdictBackend(t *testing.T, handler http.Handler) (*LlamaBackend, *httptest.Server) {
	t.Helper()
	fake := httptest.NewServer(handler)
	t.Cleanup(fake.Close)
	host, port := splitHostPort(t, fake.URL)
	b := NewLlamaBackend(Config{Host: host})
	cmd := attachPlaceholderChild(t, b, port)
	t.Cleanup(func() { cleanupPlaceholderChild(cmd) })
	b.daemonCtx = context.Background()
	return b, fake
}

// TestEnsureStartedBusyChildIsLeftAlone: the quick probe fails but an embed
// completed moments ago — EnsureStarted must treat the child as busy and
// return nil without restarting (same cmd, no cold start).
func TestEnsureStartedBusyChildIsLeftAlone(t *testing.T) {
	b, _ := newVerdictBackend(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable) // /health starved
	}))
	before := b.getCmd()

	b.noteAlive() // an embed just completed
	if err := b.EnsureStarted(); err != nil {
		t.Fatalf("EnsureStarted on a busy child: %v", err)
	}
	if b.getCmd() != before {
		t.Fatal("busy child was restarted despite a fresh proof of life")
	}
}

// TestEnsureStartedDrainRecoverySkipsRestart: no recent proof of life, but
// the child answers the generous verdict probe once its (simulated) queue
// drains — restartIfDead must skip the restart and keep the same child.
func TestEnsureStartedDrainRecoverySkipsRestart(t *testing.T) {
	var busy atomic.Bool
	busy.Store(true)
	received := make(chan struct{})
	release := make(chan struct{})
	b, _ := newVerdictBackend(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/health") {
			if busy.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			w.WriteHeader(http.StatusOK)
			return
		}
		// the in-flight embed: blocks until released, then "queue drained"
		close(received)
		<-release
		busy.Store(false)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1]}]}`))
	}))
	before := b.getCmd()

	embedDone := make(chan error, 1)
	go func() {
		_, err := b.Embed(context.Background(), "x", true)
		embedDone <- err
	}()
	// Handshake, not a sleep: the drain assertion below is only meaningful
	// once the embed actually holds its RLock inside the fake server.
	select {
	case <-received:
	case <-time.After(2 * time.Second):
		t.Fatal("embed never reached the fake llama-server")
	}

	ensureDone := make(chan error, 1)
	go func() { ensureDone <- b.EnsureStarted() }()

	// EnsureStarted must be draining (blocked on inflightMu), not killing:
	select {
	case err := <-ensureDone:
		t.Fatalf("EnsureStarted returned (%v) without waiting for the in-flight embed", err)
	case <-time.After(300 * time.Millisecond):
	}

	close(release) // queue drains; health flips to 200

	if err := <-embedDone; err != nil {
		t.Fatalf("in-flight embed was severed: %v", err)
	}
	select {
	case err := <-ensureDone:
		if err != nil {
			t.Fatalf("EnsureStarted after drain recovery: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("EnsureStarted never returned after drain")
	}
	if b.getCmd() != before {
		t.Fatal("child was restarted even though it recovered after the drain")
	}
}

// TestEnsureStartedDeadChildIsRestarted: no proof of life, health stays down
// through the drain — the child must actually be replaced. The relaunch uses
// a binary that exits immediately, so EnsureStarted surfaces the start error;
// what matters is that the dead child was removed.
func TestEnsureStartedDeadChildIsRestarted(t *testing.T) {
	b, _ := newVerdictBackend(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable) // dead: never recovers
	}))
	b.cfg.BinaryPath = "/usr/bin/false"
	before := b.getCmd()

	err := b.EnsureStarted()
	if err == nil {
		t.Fatal("EnsureStarted should surface the failed relaunch")
	}
	if cur := b.getCmd(); cur == before {
		t.Fatal("dead child was not replaced")
	}
}
