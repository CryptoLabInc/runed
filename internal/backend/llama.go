// Package backend manages the llama-server child process that runed uses
// for Qwen3-Embedding inference.
package backend

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"sync"
	"syscall"
	"time"
)

type Config struct {
	BinaryPath string
	ModelPath  string
	LogPath    string // if non-empty, stderr → file
	Host       string // default 127.0.0.1
	CtxSize    int    // default 2048; --ctx-size in tokens = max input length (llama-server rejects longer input with HTTP 400)
}

type LlamaBackend struct {
	cfg Config

	cmd  *exec.Cmd
	port int
	mu   sync.Mutex // protects cmd, port — short critical sections only

	// lifecycleMu serializes Start / Stop / EnsureStarted. RPC handlers
	// call EnsureStarted on every request so the contention here must
	// stay cheap (a healthy fast-path returns in ~ms).
	lifecycleMu sync.Mutex
	// inflightMu pairs Embed/EmbedBatch (readers) with Stop (writer).
	// Without it, the idle ticker can SIGTERM llama-server while an
	// Embed HTTP call is in flight — the call would die with a
	// connection reset. Stop takes the writer Lock so it waits for
	// every in-flight Embed to finish before signalling the child.
	inflightMu sync.RWMutex
	// daemonCtx is the long-lived context recorded on the first Start.
	// EnsureStarted re-spawns the child under this ctx instead of a
	// short-lived RPC context, so the resurrected llama-server outlives
	// the request that woke it.
	daemonCtx context.Context
}

func NewLlamaBackend(cfg Config) *LlamaBackend {
	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.CtxSize <= 0 {
		cfg.CtxSize = 2048
	}
	return &LlamaBackend{cfg: cfg}
}

func (b *LlamaBackend) Port() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.port
}

// CtxSize returns the configured context size in tokens (= max input length).
// Set once in NewLlamaBackend and never mutated, so no lock is needed.
func (b *LlamaBackend) CtxSize() int { return b.cfg.CtxSize }

// portRe matches llama-server's log line. The upstream format (observed) is:
//
//	main: server is listening on http://127.0.0.1:53183
//
// Older/alternate builds sometimes emit:
//
//	HTTP server listening on host 127.0.0.1, port 34567
//
// We accept either shape: host:port URL form, or explicit "port N" form.
var portRe = regexp.MustCompile(`(?i)listening on .*?(?::(\d+)\b|port\s+(\d+))`)

// Start launches the llama-server child and waits until it's healthy.
// The provided ctx is recorded as the daemon-lifetime context — later
// EnsureStarted calls reuse it so resurrections after idle-suspend run
// under the original lifetime, not under any short RPC ctx.
func (b *LlamaBackend) Start(ctx context.Context) error {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	b.daemonCtx = ctx
	return b.startLocked(ctx)
}

// EnsureStarted brings the backend back up if it's been stopped (idle
// suspend) or has somehow died. Idempotent — safe to call on every RPC
// entry: a healthy backend returns within milliseconds (just a quick
// /health probe under lifecycleMu).
//
// Pre-condition: Start has been called at least once so daemonCtx is
// recorded. Returns an error if the daemonCtx has been cancelled
// (process shutdown in progress).
func (b *LlamaBackend) EnsureStarted() error {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	if b.daemonCtx == nil {
		return errors.New("backend not initialized; Start must be called once first")
	}
	if err := b.daemonCtx.Err(); err != nil {
		return fmt.Errorf("backend daemon context done: %w", err)
	}

	b.mu.Lock()
	haveCmd := b.cmd != nil
	b.mu.Unlock()
	if haveCmd {
		if b.IsHealthy(b.daemonCtx) {
			return nil
		}
		slog.Warn("backend: process alive but unhealthy, restarting")
		_ = b.stopLocked(context.Background())
	}
	slog.Info("backend: cold start (resuming after suspend)")
	start := time.Now()
	if err := b.startLocked(b.daemonCtx); err != nil {
		return err
	}
	slog.Info("backend: cold start complete",
		"port", b.Port(),
		"duration_ms", time.Since(start).Milliseconds())
	return nil
}

func (b *LlamaBackend) startLocked(ctx context.Context) error {
	args := []string{
		"--model", b.cfg.ModelPath,
		"--embeddings",
		"--pooling", "last",
		"--ctx-size", strconv.Itoa(b.cfg.CtxSize),
		"--host", b.cfg.Host,
		"--port", "0", // OS-assigned
	}
	cmd := exec.CommandContext(ctx, b.cfg.BinaryPath, args...)

	// Merged stdout+stderr (llama-server writes startup info to stderr mostly)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}
	cmd.Stdout = cmd.Stderr // easier log collection

	// OS-specific child termination guarantee
	attachChildGuards(cmd)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start: %w", err)
	}
	b.mu.Lock()
	b.cmd = cmd
	b.port = 0
	b.mu.Unlock()

	var logW io.Writer = io.Discard
	var logFile *os.File
	if b.cfg.LogPath != "" {
		f, err := os.Create(b.cfg.LogPath)
		if err == nil {
			logFile = f
			logW = f
		}
	}

	portCh := make(chan int, 1)
	scannerDone := make(chan struct{})
	go func() {
		defer close(scannerDone)
		if logFile != nil {
			defer logFile.Close()
		}
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			fmt.Fprintln(logW, line)
			if m := portRe.FindStringSubmatch(line); len(m) > 0 {
				// Either capture group 1 (host:port form) or 2 ("port N" form)
				// holds the digits. Pick whichever is non-empty.
				var raw string
				for _, g := range m[1:] {
					if g != "" {
						raw = g
						break
					}
				}
				if raw == "" {
					continue
				}
				var p int
				fmt.Sscanf(raw, "%d", &p)
				select {
				case portCh <- p:
				default:
				}
			}
		}
	}()

	// Wait for port + health
	select {
	case p := <-portCh:
		b.mu.Lock()
		b.port = p
		b.mu.Unlock()
	case <-scannerDone:
		// Scanner ended before emitting a port → child likely exited early
		// (bad model, OOM, etc.). Reap cmd and surface Wait's error.
		waitErr := cmd.Wait()
		b.mu.Lock()
		b.cmd = nil
		b.port = 0
		b.mu.Unlock()
		if waitErr != nil {
			return fmt.Errorf("llama-server exited before ready: %w", waitErr)
		}
		return fmt.Errorf("llama-server exited before ready")
	case <-ctx.Done():
		_ = b.stopLocked(context.Background())
		return fmt.Errorf("timed out waiting for llama-server port")
	}

	// Health poll up to 15s
	healthCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	for {
		if b.IsHealthy(healthCtx) {
			return nil
		}
		select {
		case <-scannerDone:
			// Child exited during health polling.
			waitErr := cmd.Wait()
			b.mu.Lock()
			b.cmd = nil
			b.port = 0
			b.mu.Unlock()
			if waitErr != nil {
				return fmt.Errorf("llama-server exited during health check: %w", waitErr)
			}
			return fmt.Errorf("llama-server exited during health check")
		case <-healthCtx.Done():
			_ = b.stopLocked(context.Background())
			return fmt.Errorf("llama-server not healthy within deadline")
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// getCmd returns the currently-running command under the mutex.
// Returns nil if Start has not been called, Start failed, or Stop completed.
func (b *LlamaBackend) getCmd() *exec.Cmd {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cmd
}

func (b *LlamaBackend) IsHealthy(ctx context.Context) bool {
	port := b.Port()
	if port == 0 {
		return false
	}
	ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	url := fmt.Sprintf("http://%s:%d/health", b.cfg.Host, port)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// Stop terminates the llama-server child if running. After Stop returns,
// cmd is nil and port is 0 — a subsequent EnsureStarted re-launches the
// child cleanly.
//
// inflightMu's writer Lock is taken first so Stop blocks until every
// in-flight Embed/EmbedBatch HTTP call finishes; otherwise the idle
// ticker could SIGTERM llama-server mid-request and the client would
// see a connection reset. Lock order is inflightMu → lifecycleMu — the
// reverse never appears in any code path, so no deadlock.
func (b *LlamaBackend) Stop(ctx context.Context) error {
	b.inflightMu.Lock()
	defer b.inflightMu.Unlock()
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	return b.stopLocked(ctx)
}

func (b *LlamaBackend) stopLocked(ctx context.Context) error {
	cmd := b.getCmd()
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	clear := func() {
		b.mu.Lock()
		b.cmd = nil
		b.port = 0
		b.mu.Unlock()
	}

	select {
	case <-done:
		clear()
		return nil
	case <-time.After(5 * time.Second):
		_ = cmd.Process.Kill()
		<-done
		clear()
		return nil
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		<-done
		clear()
		return ctx.Err()
	}
}
