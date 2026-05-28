//go:build !windows

package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestIsHealthyTimesOutOnHungServer verifies the internal 500ms timeout
// fires when /health never responds. Without it, IsHealthy inherits the
// caller's (often deadline-less) ctx and blocks forever — wedging the
// idle-suspend ticker for the rest of the daemon's life.
func TestIsHealthyTimesOutOnHungServer(t *testing.T) {
	block := make(chan struct{})
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	// Cleanups run LIFO and only after deferreds, so unblock the handler
	// before fake.Close — otherwise Close hangs draining the live connection.
	t.Cleanup(fake.Close)
	t.Cleanup(func() { close(block) })

	host, port := splitHostPort(t, fake.URL)
	b := NewLlamaBackend(Config{Host: host})
	b.mu.Lock()
	b.port = port
	b.mu.Unlock()

	start := time.Now()
	ok := b.IsHealthy(context.Background())
	elapsed := time.Since(start)

	if ok {
		t.Fatalf("hung /health returned true")
	}
	if elapsed > 800*time.Millisecond {
		t.Fatalf("IsHealthy took %v — internal timeout did not fire", elapsed)
	}
}
