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
	"sync/atomic"
	"syscall"
	"time"
)

// Health-verdict tuning. A saturated single-slot llama-server can miss the
// quick per-RPC probe (its HTTP loop is starved by inference) while being
// perfectly alive — killing it then severs every queued embed and the
// restarted server saturates again immediately, a restart loop observed in
// production (dozens of kills over 3 minutes, 244 embeds cut with EOF).
const (
	// quickProbeTimeout is the cheap per-RPC /health probe. Misses under
	// saturation are expected and are NOT treated as death on their own.
	quickProbeTimeout = 500 * time.Millisecond
	// verdictProbeTimeout is the generous probe used when actually deciding
	// whether to restart.
	verdictProbeTimeout = 2 * time.Second
	// aliveGrace: if any embed completed or probe succeeded this recently,
	// the child is alive by definition — a failed quick probe means busy.
	aliveGrace = 15 * time.Second
)

type Config struct {
	BinaryPath string
	ModelPath  string
	LogPath    string // if non-empty, stderr → file
	Host       string // default 127.0.0.1
	CtxSize    int    // default 2048; --ctx-size in tokens = max input length (llama-server rejects longer input with HTTP 400)
	// PidPath, if non-empty, records the spawned llama-server's pid+binary so
	// the next runed can reap an orphan left by a SIGKILLed predecessor (see
	// pidfile.go). Empty disables recording and sweeping.
	PidPath string
}

// ServingState is the backend's current ability to serve embeddings, as
// reported through the daemon's Health RPC.
type ServingState int

const (
	// ServingOK: the llama-server child is up and answering /health.
	ServingOK ServingState = iota
	// ServingIdle: the child was intentionally stopped after an idle period
	// to free model memory. The next Embed RPC resurrects it (a one-off
	// cold-start latency). This is expected, not a fault.
	ServingIdle
	// ServingDegraded: the child is up but failing /health, or it exited
	// unexpectedly. A genuine problem.
	ServingDegraded
)

// childState records why the llama-server child is not currently running,
// so Serving can tell an intentional idle-suspend apart from a crash.
type childState int

const (
	childRunning   childState = iota // child up (or starting)
	childSuspended                   // intentionally stopped (idle suspend / restart)
	childFailed                      // exited unexpectedly
)

type LlamaBackend struct {
	cfg Config

	cmd      *exec.Cmd
	port     int
	cmdDone  chan struct{} // closed by watchChild when current cmd.Wait returns
	stopping bool          // true while stopLocked is intentionally terminating the child
	// state records why the child is down (childSuspended vs childFailed) so
	// Serving maps it to ServingIdle vs ServingDegraded. Protected by mu.
	state childState
	mu    sync.Mutex // protects cmd, port, cmdDone, stopping, state — short critical sections only

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

	// restartMu single-flights the drain-and-verdict restart path so a burst
	// of RPCs that all missed the quick probe elects one leader; followers
	// find the child verified (or restarted) and return. Lock order:
	// restartMu → inflightMu → lifecycleMu.
	restartMu sync.Mutex
	// lastAlive is the unix-nano time of the last proof of life: a successful
	// /health probe or a completed embed HTTP call. It distinguishes a busy
	// child (making progress, probes starved) from a dead one.
	lastAlive atomic.Int64
}

// noteAlive records a proof of life. Called on successful health probes and
// completed embed calls (see embed.go).
func (b *LlamaBackend) noteAlive() { b.lastAlive.Store(time.Now().UnixNano()) }

// recentlyAlive reports whether a proof of life landed within grace.
func (b *LlamaBackend) recentlyAlive(grace time.Duration) bool {
	ns := b.lastAlive.Load()
	return ns != 0 && time.Since(time.Unix(0, ns)) < grace
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
	// A previous runed that died by SIGKILL/jetsam could not run its shutdown
	// path; if its llama-server is still alive, reap it before spawning ours
	// or the machine ends up with two ~1.7 GB inference servers.
	sweepOrphanChild(b.cfg.PidPath, b.cfg.BinaryPath)
	return b.startLocked(ctx)
}

// EnsureStarted brings the backend back up if it's been stopped (idle
// suspend) or has somehow died. Idempotent — safe to call on every RPC
// entry: a healthy backend returns within milliseconds (just a quick
// /health probe under lifecycleMu).
//
// A failed quick probe on a running child is NOT treated as death: a
// saturated llama-server starves its HTTP loop and misses the probe while
// still making progress. If a proof of life (completed embed or successful
// probe) landed within aliveGrace the child is considered busy and left
// alone; only a child with no life signs goes to restartIfDead, which drains
// in-flight work and re-probes before killing anything.
//
// Pre-condition: Start has been called at least once so daemonCtx is
// recorded. Returns an error if the daemonCtx has been cancelled
// (process shutdown in progress).
func (b *LlamaBackend) EnsureStarted() error {
	b.lifecycleMu.Lock()
	if b.daemonCtx == nil {
		b.lifecycleMu.Unlock()
		return errors.New("backend not initialized; Start must be called once first")
	}
	dctx := b.daemonCtx
	if err := dctx.Err(); err != nil {
		b.lifecycleMu.Unlock()
		return fmt.Errorf("backend daemon context done: %w", err)
	}

	b.mu.Lock()
	haveCmd := b.cmd != nil
	b.mu.Unlock()

	if !haveCmd { // intentional idle-suspend (or a crash) — plain cold start
		slog.Info("backend: cold start (resuming after suspend)")
		start := time.Now()
		err := b.startLocked(dctx)
		if err == nil {
			slog.Info("backend: cold start complete",
				"port", b.Port(),
				"duration_ms", time.Since(start).Milliseconds())
		}
		b.lifecycleMu.Unlock()
		return err
	}

	if b.probeHealthy(dctx, quickProbeTimeout) {
		b.lifecycleMu.Unlock()
		return nil
	}
	if b.recentlyAlive(aliveGrace) {
		// Probe starved but work is completing — saturated, not dead.
		b.lifecycleMu.Unlock()
		return nil
	}
	b.lifecycleMu.Unlock()
	// No proof of life within grace: escalate — but restartIfDead still
	// drains and re-probes before it kills anything.
	return b.restartIfDead(dctx)
}

// restartIfDead is the only path that may kill an unresponsive child. It
// tells busy apart from dead by construction:
//
//  1. restartMu single-flights bursts of suspicious RPCs;
//  2. a generous re-probe catches a child a predecessor already fixed;
//  3. the inflightMu writer lock DRAINS every in-flight embed — a saturated
//     child empties its queue right here (and nothing gets severed with EOF,
//     which the old path did by skipping this lock);
//  4. a post-drain re-probe: a child that answers now was busy — restart is
//     skipped. Only a child that stays unresponsive with zero load is dead.
func (b *LlamaBackend) restartIfDead(dctx context.Context) error {
	b.restartMu.Lock()
	defer b.restartMu.Unlock()

	// Followers queued behind a leader re-check the cheap signals first: the
	// leader's drain let embeds complete (stamping lastAlive) and a restart's
	// startup health poll stamps it too — without this, every queued follower
	// would run its own drain cycle in series.
	if b.recentlyAlive(aliveGrace) {
		return nil
	}
	if b.probeHealthy(dctx, verdictProbeTimeout) {
		return nil // recovered (or a predecessor restarted it)
	}

	b.inflightMu.Lock()
	defer b.inflightMu.Unlock()
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()

	b.mu.Lock()
	haveCmd := b.cmd != nil
	b.mu.Unlock()
	if haveCmd && b.probeHealthy(dctx, verdictProbeTimeout) {
		slog.Info("backend: health recovered after draining in-flight work — busy, not dead; restart skipped")
		return nil
	}

	if haveCmd {
		slog.Warn("backend: unresponsive with no in-flight work, restarting")
		_ = b.stopLocked(context.Background())
	}
	slog.Info("backend: cold start (restart after failed health verdict)")
	start := time.Now()
	if err := b.startLocked(dctx); err != nil {
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
		// Embedding-only workload: every prompt is unique, so the prompt cache
		// (default cap 8 GiB) grows ~14 MiB per request at a ~0% hit rate —
		// disable it. Likewise skip REPACK's private anonymous copy of the
		// weights so they stay on the shared, reclaimable mmap (-640 MiB
		// resident, ~+20% embed latency on CPU). One slot instead of auto(4):
		// embeds are short one-shot requests, and 4 slots quadruple the KV
		// cache (~940 MiB at ctx 2048) for concurrency this workload lacks.
		"--cache-ram", "0",
		"--no-repack",
		"--parallel", "1",
		// Smaller physical batch + quantized KV: compute scratch scales with
		// ubatch and the KV with ctx×precision; neither changes results or the
		// max input length, only peak memory (and adds modest latency).
		"--batch-size", "256",
		"--ubatch-size", "256",
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
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
	// Watcher owns cmd.Wait — exactly one caller, no zombies, and unexpected
	// exits get logged + state cleared so EnsureStarted re-spawns cleanly
	// instead of carrying a stale b.cmd pointing at a dead PID.
	done := make(chan struct{})
	b.mu.Lock()
	b.cmd = cmd
	b.port = 0
	b.cmdDone = done
	b.state = childRunning
	// Record the child before anything can fail below: if runed itself is
	// SIGKILLed from here on, the next boot's sweep finds the orphan. The
	// write happens under mu, in the same critical section that installs
	// b.cmd, so it is totally ordered against watchChild's owner-guarded
	// clear — otherwise a watcher reaping the PREVIOUS child could delete
	// this record right after we write it.
	if err := writeChildPid(b.cfg.PidPath, cmd.Process.Pid, b.cfg.BinaryPath); err != nil {
		slog.Warn("backend: could not record llama pidfile (orphan sweep degraded)", "err", err)
	}
	b.mu.Unlock()
	go b.watchChild(cmd, done)

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

	// Wait for port + health. The watcher (not us) calls cmd.Wait, so error
	// paths here wait on done instead of double-calling Wait. The watcher
	// will have logged the underlying exit error via slog.Warn.
	select {
	case p := <-portCh:
		b.mu.Lock()
		b.port = p
		b.mu.Unlock()
	case <-scannerDone:
		// Scanner ended before emitting a port → child exited early
		// (bad model, OOM, etc.). Watcher will clear b.cmd.
		<-done
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
			<-done
			return fmt.Errorf("llama-server exited during health check")
		case <-healthCtx.Done():
			_ = b.stopLocked(context.Background())
			return fmt.Errorf("llama-server not healthy within deadline")
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// watchChild owns the single cmd.Wait() call for cmd. It reaps the child
// immediately on exit (no zombie), clears b.cmd/b.port if they still point
// to this cmd (so EnsureStarted re-spawns instead of probing a dead PID),
// then closes done so blocked stopLocked / startLocked callers can proceed.
// Logs a warning on unexpected exits — exits caused by an in-progress
// stopLocked are silent (b.stopping == true).
func (b *LlamaBackend) watchChild(cmd *exec.Cmd, done chan struct{}) {
	err := cmd.Wait()
	b.mu.Lock()
	intentional := b.stopping
	if b.cmd == cmd {
		b.cmd = nil
		b.port = 0
		// Any exit leaves the child down; default to failed so a crash or a
		// failed (re)start reports DEGRADED. Stop() (the idle-suspend path)
		// overrides to childSuspended afterward — it waits on done, so its
		// write lands strictly after this one.
		b.state = childFailed
		// The child is reaped — drop its record so the pidfile exists
		// exactly while a child runs. Owner-guarded and under mu: only the
		// watcher of the CURRENT child clears, and the clear is totally
		// ordered against the next spawn's write (also under mu), so it can
		// never delete a successor's record.
		clearChildPid(b.cfg.PidPath)
	}
	b.mu.Unlock()
	// Log before close(done) so readers of done can rely on the log
	// already being written — avoids a test-visible race and makes the
	// ordering "child reaped → logged → others notified".
	if !intentional && err != nil {
		slog.Warn("backend: llama-server exited unexpectedly", "err", err)
	}
	close(done)
}

// getCmd returns the currently-running command under the mutex.
// Returns nil if Start has not been called, Start failed, or Stop completed.
func (b *LlamaBackend) getCmd() *exec.Cmd {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cmd
}

func (b *LlamaBackend) IsHealthy(ctx context.Context) bool {
	return b.probeHealthy(ctx, quickProbeTimeout)
}

// probeHealthy GETs /health with the given budget. A success is a proof of
// life and refreshes lastAlive.
func (b *LlamaBackend) probeHealthy(ctx context.Context, timeout time.Duration) bool {
	port := b.Port()
	if port == 0 {
		return false
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
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
	if resp.StatusCode != 200 {
		return false
	}
	b.noteAlive()
	return true
}

// Serving reports whether the backend can currently serve embeddings, and if
// not, why. Health maps the result onto the proto Status enum:
// ServingOK→OK, ServingIdle→IDLE (intentional idle-suspend, resumes on the
// next RPC), ServingDegraded→DEGRADED (process up-but-unhealthy, or crashed).
func (b *LlamaBackend) Serving(ctx context.Context) ServingState {
	b.mu.Lock()
	haveCmd := b.cmd != nil
	st := b.state
	b.mu.Unlock()
	if haveCmd {
		if b.IsHealthy(ctx) {
			return ServingOK
		}
		if b.recentlyAlive(aliveGrace) {
			// Probe starved by a saturated child that is still completing
			// work — it IS serving, just busy. Reporting DEGRADED here made
			// dashboards cry wolf during every heavy embed burst.
			return ServingOK
		}
		return ServingDegraded
	}
	if st == childSuspended {
		return ServingIdle
	}
	return ServingDegraded
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
	b.mu.Lock()
	wasRunning := b.cmd != nil
	b.mu.Unlock()
	err := b.stopLocked(ctx)
	// Stop is the intentional idle-suspend path (the idle ticker). Mark the
	// child suspended so Serving reports IDLE, not DEGRADED — but only if we
	// actually stopped a running child, so a Stop on an already-crashed
	// backend doesn't mask the failure. stopLocked waits on done, so
	// watchChild's childFailed write has already landed; this override runs
	// after it.
	if wasRunning {
		b.mu.Lock()
		b.state = childSuspended
		b.mu.Unlock()
	}
	return err
}

func (b *LlamaBackend) stopLocked(ctx context.Context) error {
	b.mu.Lock()
	cmd := b.cmd
	done := b.cmdDone
	if cmd == nil || cmd.Process == nil {
		b.mu.Unlock()
		return nil
	}
	// stopping=true tells watchChild this exit is intentional → no warning log.
	// Reset on return so a later spontaneous crash still gets logged.
	b.stopping = true
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.stopping = false
		b.mu.Unlock()
	}()

	_ = cmd.Process.Signal(syscall.SIGTERM)

	select {
	case <-done:
		return nil
	case <-time.After(5 * time.Second):
		_ = cmd.Process.Kill()
		<-done
		return nil
	case <-ctx.Done():
		_ = cmd.Process.Kill()
		<-done
		return ctx.Err()
	}
}
