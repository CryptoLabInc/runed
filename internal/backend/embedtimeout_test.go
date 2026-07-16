//go:build !windows

package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestStuckServerDoesNotWedgeDrain reproduces the deadlock a hung (not crashed)
// llama-server could cause: an in-flight Embed holds inflightMu.RLock while its
// HTTP call blocks, and Stop() (same inflightMu.Lock the recovery path takes)
// would then wait forever. With embedRequestTimeout bounding the call, the
// embed fails, releases the RLock, and the drain completes.
func TestStuckServerDoesNotWedgeDrain(t *testing.T) {
	stop := make(chan struct{})
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done(): // client (embedRequestTimeout) cancelled
		case <-stop: // test teardown
		}
	}))
	defer fake.Close()  // runs last — after the handler is unblocked
	defer close(stop)   // runs first — lets any stuck handler return

	orig := embedRequestTimeout
	embedRequestTimeout = 300 * time.Millisecond
	defer func() { embedRequestTimeout = orig }()

	host, port := splitHostPort(t, fake.URL)
	b := NewLlamaBackend(Config{Host: host})
	cmd := attachPlaceholderChild(t, b, port)
	defer cleanupPlaceholderChild(cmd)

	embedErr := make(chan error, 1)
	start := time.Now()
	go func() {
		_, err := b.Embed(context.Background(), "hello", true) // no caller deadline
		embedErr <- err
	}()

	// Must fail on its own timeout, not hang forever.
	select {
	case err := <-embedErr:
		if err == nil {
			t.Fatal("embed against a hung server returned nil; expected a timeout error")
		}
		if d := time.Since(start); d > 3*time.Second {
			t.Fatalf("embed took %v; embedRequestTimeout not bounding the call", d)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("embed did not return — embedRequestTimeout is not bounding the HTTP call")
	}

	// The drain (inflightMu.Lock, as restartIfDead does) must now complete.
	drained := make(chan struct{})
	go func() {
		b.inflightMu.Lock()
		b.inflightMu.Unlock()
		close(drained)
	}()
	select {
	case <-drained:
	case <-time.After(2 * time.Second):
		t.Fatal("inflightMu never drained — a stuck embed still wedges the recovery path")
	}
}
