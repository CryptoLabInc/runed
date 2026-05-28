//go:build !windows

package backend

import (
	"bytes"
	"context"
	"log/slog"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestWatcherClearsStateOnUnexpectedExit verifies that when the child dies
// on its own (no Stop), the watcher reaps it, clears b.cmd/b.port, and
// closes the done channel — so the next EnsureStarted sees a clean slate
// instead of probing a dead PID.
func TestWatcherClearsStateOnUnexpectedExit(t *testing.T) {
	b := NewLlamaBackend(Config{})

	cmd := exec.Command("sleep", "60")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	done := make(chan struct{})
	b.mu.Lock()
	b.cmd = cmd
	b.port = 12345
	b.cmdDone = done
	b.mu.Unlock()
	go b.watchChild(cmd, done)

	// Simulate unexpected death — SIGKILL bypasses any cleanup the child
	// might do and is what an OOM-killer would use.
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("kill: %v", err)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("watcher did not close done after child died")
	}

	b.mu.Lock()
	gotCmd, gotPort := b.cmd, b.port
	b.mu.Unlock()
	if gotCmd != nil {
		t.Errorf("b.cmd not cleared after unexpected exit: %v", gotCmd)
	}
	if gotPort != 0 {
		t.Errorf("b.port not cleared after unexpected exit: %d", gotPort)
	}
}

// TestWatcherSilentOnIntentionalStop verifies that Stop's SIGTERM does NOT
// produce an "exited unexpectedly" warning — that log noise would mask
// genuine crashes.
func TestWatcherSilentOnIntentionalStop(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	b := NewLlamaBackend(Config{})
	cmd := attachPlaceholderChild(t, b, 1)
	defer cleanupPlaceholderChild(cmd)

	if err := b.Stop(context.Background()); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	// Watcher closes done before its log decision, so by Stop's return the
	// log state is settled.
	if strings.Contains(buf.String(), "exited unexpectedly") {
		t.Fatalf("intentional Stop produced unexpected-exit warning:\n%s", buf.String())
	}
}

// TestWatcherLogsUnexpectedExit is the positive complement: when the child
// dies WITHOUT Stop running, the warning fires so operators see the crash.
func TestWatcherLogsUnexpectedExit(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	b := NewLlamaBackend(Config{})

	cmd := exec.Command("sleep", "60")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn: %v", err)
	}
	done := make(chan struct{})
	b.mu.Lock()
	b.cmd = cmd
	b.cmdDone = done
	b.mu.Unlock()
	go b.watchChild(cmd, done)

	_ = cmd.Process.Kill()
	<-done

	if !strings.Contains(buf.String(), "exited unexpectedly") {
		t.Fatalf("crash-style exit did not produce warning:\n%s", buf.String())
	}
}
