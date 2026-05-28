// Command runed is the embedding daemon.
//
// Self-bootstrap: when RUNED_LLAMA_SERVER or RUNED_MODEL is unset, runed
// downloads the llama-server release tarball and embedding GGUF described
// by the manifest at RUNED_MANIFEST (or the build-time DefaultManifestURL)
// into $RUNED_HOME/{bin/llama-cpp,models}/ on first boot. Subsequent boots
// reuse the installed artifacts as long as the manifest SHA-256s still
// match. To bypass self-bootstrap entirely, set both RUNED_LLAMA_SERVER
// and RUNED_MODEL to absolute paths.
//
// The gRPC UDS opens before self-bootstrap so clients can dial immediately
// and poll Health for STATUS_LOADING + Phase/bytes progress instead of
// seeing dial failures during the multi-minute install window. Embed and
// EmbedBatch return FAILED_PRECONDITION until the backend is wired.
//
// Optional environment:
//
//	RUNED_HOME            Data directory (default: $HOME/.runed).
//	RUNED_CTX_SIZE        Max input length in tokens; lower = less KV-cache memory (default: 2048).
//	RUNED_MODEL_VARIANT   Override manifest.default_model (e.g. "qwen3-embedding-0.6b.q8_0").
//	RUNED_MANIFEST        Manifest URL for self-bootstrap. Overrides DefaultManifestURL baked at build.
//	RUNED_IDLE_TIMEOUT    Idle duration after which llama-server is stopped to
//	                      release model weights from memory. runed itself
//	                      stays up; the next Embed/EmbedBatch RPC resurrects
//	                      the backend via backend.EnsureStarted (cold-start
//	                      latency paid only on that first post-idle request).
//	                      Default 10m; "0" disables backend suspend.
//
// The daemon listens on $RUNED_HOME/embedding.sock (UDS) and terminates
// gracefully on SIGINT, SIGTERM, or a Shutdown RPC. Graceful termination
// drains in-flight gRPC calls (10s ceiling) and then stops the llama-server
// child process; the UDS file is auto-unlinked when the listener closes.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"github.com/CryptoLabInc/runed/internal/backend"
	"github.com/CryptoLabInc/runed/internal/bootstrap"
	"github.com/CryptoLabInc/runed/internal/ipc"
	"github.com/CryptoLabInc/runed/internal/server"
	"google.golang.org/grpc"
)

// daemonVersion is set at build time via -ldflags "-X main.daemonVersion=..."
// The default is a development marker so `go run` / un-flagged builds still
// produce a sensible Info.daemon_version.
var daemonVersion = "v0.1.0-alpha"

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	paths, err := bootstrap.Resolve()
	if err != nil {
		return fmt.Errorf("resolve paths: %w", err)
	}
	if err := paths.EnsureDirs(); err != nil {
		return fmt.Errorf("ensure dirs: %w", err)
	}
	logger.Info("paths resolved", "home", paths.Home)

	sockPath := filepath.Join(paths.Home, "embedding.sock")

	// Early bail-out: if another daemon is already accepting connections on
	// our socket, we'd just waste time on self-bootstrap before failing at
	// the listen step. Exit 0 because "already running" is a no-op success
	// from the caller's perspective (rune spawn, systemd restart, etc.).
	if anotherDaemonReachable(ctx, sockPath) {
		logger.Info("another runed daemon is already serving this socket; exiting",
			"socket", sockPath)
		return nil
	}
	// Stale socket file (no listener) — remove so our own listen succeeds.
	// This happens after SIGKILL / OOM where graceful shutdown didn't run.
	if _, err := os.Stat(sockPath); err == nil {
		logger.Info("removing stale socket file", "socket", sockPath)
		if err := os.Remove(sockPath); err != nil {
			return fmt.Errorf("remove stale socket %s: %w", sockPath, err)
		}
	}

	// ctx-size: RUNED_CTX_SIZE if valid, else backend default (2048).
	ctxSize := 0
	if v := os.Getenv("RUNED_CTX_SIZE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			ctxSize = n
		} else {
			logger.Warn("invalid RUNED_CTX_SIZE, using default", "value", v)
		}
	}

	// Listen before self-bootstrap so clients can poll Health for
	// STATUS_LOADING + Phase/progress instead of dial failures during
	// the multi-minute install window. SetBackend below flips Health
	// to STATUS_OK once llama-server is up.
	srv := server.New(daemonVersion)
	lis, err := ipc.Listen(sockPath)
	if err != nil {
		return fmt.Errorf("ipc listen: %w", err)
	}
	logger.Info("listening", "socket", sockPath)

	gs := grpc.NewServer(grpc.UnaryInterceptor(srv.UnaryActivityInterceptor()))
	runedv1.RegisterRunedServiceServer(gs, srv)

	// Serve in a goroutine so main can block on signals/Shutdown/serve error.
	// gs.Serve returns nil on graceful stop; any non-nil err is a real fault.
	serveErr := make(chan error, 1)
	go func() {
		if err := gs.Serve(lis); err != nil {
			serveErr <- err
		}
		close(serveErr)
	}()

	// Forward bootstrap progress into Health via stage→Phase mapping.
	reporter := bootstrap.StatusReporter(func(stage string, done, total int64) {
		phase, msg := stagePhase(stage)
		srv.SetBootstrapStatus(phase, done, total, msg)
	})

	llamaBin := os.Getenv("RUNED_LLAMA_SERVER")
	model := os.Getenv("RUNED_MODEL")
	if llamaBin == "" || model == "" {
		// Manifest fetch has no dedicated Phase enum; use UNSPECIFIED and
		// convey the stage through message. selfBootstrap overwrites
		// Phase on entering ensure{LlamaServer,Model}.
		srv.SetBootstrapStatus(runedv1.HealthResponse_PHASE_UNSPECIFIED, 0, 0, "fetching manifest")
		bin, mp, err := selfBootstrap(ctx, logger, paths, llamaBin == "", model == "", reporter)
		if err != nil {
			bailBoot(logger, srv, gs, nil)
			return err
		}
		if llamaBin == "" {
			llamaBin = bin
		}
		if model == "" {
			model = mp
		}
	} else {
		logger.Info("skipping self-bootstrap; both env paths provided",
			"llama_server", llamaBin, "model", model)
	}

	srv.SetBootstrapStatus(runedv1.HealthResponse_PHASE_STARTING_LLAMA_SERVER, 0, 0, "starting llama-server")

	modelID, err := sha256File(model)
	if err != nil {
		bailBoot(logger, srv, gs, nil)
		return fmt.Errorf("model hash: %w", err)
	}
	logger.Info("model identity", "sha256", modelID, "path", model)

	logger.Info("starting llama-server", "binary", llamaBin, "model", model)
	b := backend.NewLlamaBackend(backend.Config{
		BinaryPath: llamaBin,
		ModelPath:  model,
		LogPath:    filepath.Join(paths.Logs, "llama-server.log"),
		CtxSize:    ctxSize,
	})
	// NOTE: backend uses exec.CommandContext(ctx, ...) internally, which means
	// the child llama-server dies when this ctx is Done. We therefore pass the
	// long-lived daemon ctx rather than a short-lived timeout wrapper —
	// backend.Start already bounds start-up on its own (~15s health poll +
	// early-exit detection via the stderr scanner).
	if err := b.Start(ctx); err != nil {
		// b.Start may have spawned a child that failed health-probe;
		// bailBoot's b.Stop reaps it.
		bailBoot(logger, srv, gs, b)
		return fmt.Errorf("backend start: %w", err)
	}
	logger.Info("llama-server ready", "port", b.Port())

	// Flip Health to STATUS_OK and accept Embed/EmbedBatch.
	srv.SetBackend(b, modelID)

	idleTimeout, err := parseIdleTimeout()
	if err != nil {
		bailBoot(logger, srv, gs, b)
		return fmt.Errorf("RUNED_IDLE_TIMEOUT: %w", err)
	}
	if idleTimeout > 0 {
		logger.Info("idle backend-suspend enabled", "timeout", idleTimeout.String())
		go func() {
			// Idle policy: when no RPC arrives for RUNED_IDLE_TIMEOUT, stop the
			// llama-server child to release its memory (~470MB+ of model
			// weights). runed itself stays up so the gRPC socket remains
			// reachable — the next Embed/EmbedBatch RPC triggers
			// backend.EnsureStarted which re-spawns llama-server, paying a
			// cold-start latency hit (~hundreds of ms) only on that first
			// post-idle request. Ticker cadence is 30s, so observed suspend
			// latency is RUNED_IDLE_TIMEOUT + up to 30s.
			ticker := time.NewTicker(30 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					elapsed := time.Since(srv.LastActivity())
					if elapsed <= idleTimeout {
						continue
					}
					if !b.IsHealthy(ctx) {
						// Already suspended; nothing to do.
						continue
					}
					logger.Info("idle backend-suspend: stopping llama-server",
						"elapsed", elapsed.String())
					if err := b.Stop(ctx); err != nil {
						logger.Warn("backend stop failed", "err", err)
					}
				}
			}
		}()
	} else {
		logger.Info("idle backend-suspend disabled (RUNED_IDLE_TIMEOUT=0)")
	}

	sigCh := make(chan os.Signal, 1)
	// SIGHUP is included so closing the controlling terminal triggers the
	// graceful shutdown path; otherwise Go's default SIGHUP handler kills
	// runed without running b.Stop(), orphaning the llama-server child.
	// SIGKILL cannot be intercepted from user space.
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)

	select {
	case s := <-sigCh:
		logger.Info("received signal", "signal", s.String())
	case <-srv.ShutdownCh():
		logger.Info("received Shutdown RPC")
	case err := <-serveErr:
		if err != nil {
			logger.Error("gRPC serve error", "err", err)
			// Still fall through to backend cleanup below.
			_ = b.Stop(context.Background())
			return fmt.Errorf("serve: %w", err)
		}
	}

	// Flip Health to STATUS_SHUTTING_DOWN for all exit triggers (OS
	// signal / serve error / Shutdown RPC). sync.Once makes a redundant
	// call a no-op.
	srv.TriggerShutdown()

	// Phase 1: drain in-flight RPCs. GracefulStop blocks until all active
	// handlers return; 10s is a safety net for a wedged client.
	logger.Info("draining in-flight requests")
	graceDone := make(chan struct{})
	go func() {
		gs.GracefulStop()
		close(graceDone)
	}()
	select {
	case <-graceDone:
		logger.Info("graceful stop complete")
	case <-time.After(10 * time.Second):
		logger.Warn("graceful stop timed out, forcing")
		gs.Stop()
		<-graceDone
	}

	// Phase 2: stop backend. Deferred until after GracefulStop because Embed
	// handlers may still be waiting on backend HTTP calls during drain.
	if err := b.Stop(context.Background()); err != nil {
		logger.Warn("backend stop returned error", "err", err)
	}
	logger.Info("shutdown complete")
	return nil
}

// selfBootstrap fetches the manifest and installs the side(s) the
// caller needs. needLlama / needModel reflect which env paths are
// unset; only those sides are downloaded so an env override doesn't
// pay for an artifact it will discard. Both needed → EnsureAll's
// single-lock fast path.
func selfBootstrap(ctx context.Context, logger *slog.Logger, paths *bootstrap.Paths, needLlama, needModel bool, reporter bootstrap.StatusReporter) (binPath, modelPath string, err error) {
	manifestURL := bootstrap.ResolveManifestURL()
	if manifestURL == "" {
		return "", "", fmt.Errorf(
			"self-bootstrap needed (RUNED_LLAMA_SERVER or RUNED_MODEL unset) but no manifest URL: set %s or rebuild with DEFAULT_MANIFEST_URL",
			bootstrap.EnvManifest)
	}
	// A MITM that rewrites the manifest can also rewrite the SHA256s,
	// so artifact integrity collapses to "trust the manifest channel."
	if !strings.HasPrefix(manifestURL, "https://") {
		logger.Warn("manifest URL is not HTTPS; artifact integrity rests on manifest channel trust",
			"url", manifestURL)
	}
	logger.Info("self-bootstrap: fetching manifest", "url", manifestURL)
	mani, err := bootstrap.FetchManifest(ctx)
	if err != nil {
		return "", "", fmt.Errorf("bootstrap manifest: %w", err)
	}
	logger.Info("self-bootstrap: manifest parsed",
		"platforms", len(mani.Platforms),
		"models", len(mani.Models),
		"default_model", mani.DefaultModel)
	switch {
	case needLlama && needModel:
		var variant string
		binPath, modelPath, variant, err = bootstrap.EnsureAll(ctx, paths, mani, logger, reporter)
		if err != nil {
			return "", "", fmt.Errorf("bootstrap install: %w", err)
		}
		logger.Info("self-bootstrap ready",
			"llama_server", binPath,
			"model", modelPath,
			"variant", variant)
	case needLlama:
		binPath, err = bootstrap.EnsureLlamaServer(ctx, paths, mani, logger, reporter)
		if err != nil {
			return "", "", fmt.Errorf("bootstrap install: %w", err)
		}
		logger.Info("self-bootstrap ready (llama-server only; model from env)",
			"llama_server", binPath)
	case needModel:
		var variant string
		modelPath, variant, err = bootstrap.EnsureModel(ctx, paths, mani, logger, reporter)
		if err != nil {
			return "", "", fmt.Errorf("bootstrap install: %w", err)
		}
		logger.Info("self-bootstrap ready (model only; llama-server from env)",
			"model", modelPath,
			"variant", variant)
	default:
		return "", "", fmt.Errorf("selfBootstrap called with nothing to do")
	}
	return binPath, modelPath, nil
}

// stagePhase maps a bootstrap stage name to its proto Phase + message.
// PHASE_UNSPECIFIED on unknown stages keeps the surface forward-
// compatible with future stages.
func stagePhase(stage string) (runedv1.HealthResponse_Phase, string) {
	switch stage {
	case "llama_server":
		return runedv1.HealthResponse_PHASE_FETCHING_LLAMA_SERVER, "fetching llama-server"
	case "model":
		return runedv1.HealthResponse_PHASE_FETCHING_MODEL, "fetching embedding model"
	default:
		return runedv1.HealthResponse_PHASE_UNSPECIFIED, ""
	}
}

// bailBoot runs the normal-exit cleanup sequence for boot-time failures
// so clients see one final STATUS_SHUTTING_DOWN instead of a sudden
// disconnect. b may be nil; b.Stop is idempotent.
func bailBoot(logger *slog.Logger, srv *server.Server, gs *grpc.Server, b *backend.LlamaBackend) {
	srv.TriggerShutdown()
	stopGRPC(logger, gs)
	if b != nil {
		_ = b.Stop(context.Background())
	}
}

// stopGRPC drives a 10s-bounded GracefulStop with force-Stop fallback.
func stopGRPC(logger *slog.Logger, gs *grpc.Server) {
	graceDone := make(chan struct{})
	go func() {
		gs.GracefulStop()
		close(graceDone)
	}()
	select {
	case <-graceDone:
	case <-time.After(10 * time.Second):
		logger.Warn("graceful stop timed out, forcing")
		gs.Stop()
		<-graceDone
	}
}

// anotherDaemonReachable returns true if a UDS listener accepts our dial
// at sockPath within 500ms. False covers both "file missing" and "file
// exists but nobody listening" (the latter is a stale-socket case the
// caller should clean up before its own listen).
func anotherDaemonReachable(ctx context.Context, sockPath string) bool {
	cctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	var d net.Dialer
	conn, err := d.DialContext(cctx, "unix", sockPath)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// sha256File returns a short, prefixed SHA-256 identifier of the file at path.
// The 16-char hex truncation keeps Info.model_identity compact while retaining
// enough entropy to distinguish GGUF revisions in practice (Plan A ships a
// single model; this mostly guards against silent file swaps).
func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))[:16], nil
}

// parseIdleTimeout reads RUNED_IDLE_TIMEOUT and returns the parsed duration.
// Empty or unset env returns the default 10 minutes. A value of "0" disables
// idle exit entirely (preserves Plan A semantics). Invalid values return an
// error so the daemon refuses to start with a misconfigured timeout.
func parseIdleTimeout() (time.Duration, error) {
	raw := os.Getenv("RUNED_IDLE_TIMEOUT")
	if raw == "" {
		return 10 * time.Minute, nil
	}
	return time.ParseDuration(raw)
}
