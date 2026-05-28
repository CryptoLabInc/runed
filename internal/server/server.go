// Package server implements the gRPC RunedService by delegating to
// a LlamaBackend HTTP client.
package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"github.com/CryptoLabInc/runed/internal/backend"
	"google.golang.org/grpc"
)

// Plan A constants for the Info RPC. Qwen3-Embedding-0.6B fixes these at
// model-load time; future revisions will source them from config or the
// model file. Hardcoded here because Plan A ships with exactly one model.
const (
	vectorDim    int32 = 1024
	maxBatchSize int32 = 32
)

// Server implements runedv1.RunedServiceServer. It does not own the backend —
// callers (cmd/runed) are responsible for Start/Stop on the LlamaBackend.
type Server struct {
	runedv1.UnimplementedRunedServiceServer
	backend       *backend.LlamaBackend
	version       string
	modelIdentity string
	startedAt     time.Time

	// maxTextLength (chars) is snapshotted from the backend's ctx-size (tokens)
	// in New(); chars==tokens is conservative (dense text is ≥~1.27 chars/token),
	// so it always fits ctx. Advertised via Info → clients cap to whatever ctx
	// the daemon booted with, keeping the char limit locked to the token limit.
	maxTextLength int32

	// requests counts Embed + EmbedBatch calls (post-entry, pre-return).
	// Exposed through HealthResponse.total_requests so clients can observe
	// daemon throughput without scraping logs.
	requests atomic.Int64

	// shutdownOnce guarantees close(shutdownCh) runs exactly once even under
	// a flurry of concurrent Shutdown RPCs (double-close panics).
	shutdownOnce sync.Once
	shutdownCh   chan struct{}

	// lastActivity records the UnixNano timestamp of the most recent RPC
	// entry (set by UnaryActivityInterceptor). Used by the idle-exit ticker
	// in cmd/runed to decide when to call TriggerShutdown.
	lastActivity atomic.Int64
}

// New returns a Server that delegates Embed/EmbedBatch to backend and fills
// Info metadata from the given version and modelIdentity. max_text_length is
// snapshotted from the backend's ctx-size here (see Server.maxTextLength).
func New(b *backend.LlamaBackend, version, modelIdentity string) *Server {
	s := &Server{
		backend:       b,
		version:       version,
		modelIdentity: modelIdentity,
		startedAt:     time.Now(),
		maxTextLength: int32(b.CtxSize()),
		shutdownCh:    make(chan struct{}),
	}
	s.lastActivity.Store(time.Now().UnixNano())
	return s
}

// ShutdownCh returns a channel that closes when a Shutdown RPC is received.
// The daemon main() selects on this alongside OS signals to trigger graceful
// termination; the channel is never sent on — only closed.
func (s *Server) ShutdownCh() <-chan struct{} { return s.shutdownCh }

// LastActivity returns the time of the most recent RPC entry.
// Used by the idle-exit ticker in cmd/runed.
func (s *Server) LastActivity() time.Time {
	return time.Unix(0, s.lastActivity.Load())
}

// TriggerShutdown initiates graceful termination from inside the daemon
// (e.g. from the idle-exit ticker). Sharing shutdownOnce with the Shutdown
// RPC guarantees close(shutdownCh) runs exactly once across both triggers.
func (s *Server) TriggerShutdown() {
	s.shutdownOnce.Do(func() { close(s.shutdownCh) })
}

// UnaryActivityInterceptor returns a gRPC unary server interceptor that
// records the entry time of every RPC into lastActivity. Wired in
// cmd/runed/main.go via grpc.UnaryInterceptor.
//
// All RPCs count — including Health and Info — so a monitoring tool that
// polls Health intentionally keeps the daemon alive. This is the
// "all RPCs as activity" decision from the Plan B design doc §5.
//
// Embed-class RPCs additionally emit start/done log lines (with duration
// and error if any). Control-plane RPCs (Health/Info/Shutdown) stay
// silent here to avoid drowning the daemon log under monitor polling.
func (s *Server) UnaryActivityInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{},
		info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		s.lastActivity.Store(time.Now().UnixNano())
		if !isEmbedMethod(info.FullMethod) {
			return handler(ctx, req)
		}
		method := shortMethod(info.FullMethod)
		slog.Info("rpc start", "method", method)
		start := time.Now()
		resp, err := handler(ctx, req)
		attrs := []any{"method", method, "duration_ms", time.Since(start).Milliseconds()}
		if err != nil {
			attrs = append(attrs, "err", err.Error())
		}
		slog.Info("rpc done", attrs...)
		return resp, err
	}
}

// isEmbedMethod returns true for Embed and EmbedBatch full-method paths.
// Used by UnaryActivityInterceptor to scope per-RPC logging.
func isEmbedMethod(fullMethod string) bool {
	return strings.HasSuffix(fullMethod, "/Embed") ||
		strings.HasSuffix(fullMethod, "/EmbedBatch")
}

// shortMethod returns the trailing segment of a gRPC full-method path
// (e.g. "/runed.v1.RunedService/Embed" → "Embed"). Used for compact log
// labels.
func shortMethod(fullMethod string) string {
	if i := strings.LastIndex(fullMethod, "/"); i >= 0 {
		return fullMethod[i+1:]
	}
	return fullMethod
}

// embedMaxAttempts bounds the EnsureStarted/Embed retry loop in
// Embed/EmbedBatch. One retry covers the residual race where Stop slips
// in between EnsureStarted returning and the Embed RLock — the second
// EnsureStarted re-spawns llama-server and the second Embed proceeds
// under a fresh RLock. Bounded at 2 so a genuinely broken backend can't
// loop forever.
const embedMaxAttempts = 2

// Embed delegates to the backend's single-text embedding path.
// The proto dropped the normalize field (see commit 816ef81); the backend is
// called with normalize=true as a harmless default since llama-server always
// returns L2-normalized vectors anyway.
//
// Backend may be suspended (idle-suspend) when this RPC arrives. EnsureStarted
// resurrects it under the daemon-lifetime context — the first request after a
// suspend pays the llama-server cold-start latency (~hundreds of ms to a few
// seconds for model load); subsequent requests fall through the cheap health-
// probe fast path.
//
// Retry loop: backend.Embed holds inflightMu.RLock so Stop can't kill an
// in-flight HTTP. The remaining race window is EnsureStarted-return →
// RLock-acquire; if Stop slips into that gap we get ErrNotStarted on the
// first attempt and recover by re-running EnsureStarted once.
func (s *Server) Embed(ctx context.Context, req *runedv1.EmbedRequest) (*runedv1.EmbedResponse, error) {
	s.requests.Add(1)
	for attempt := 0; attempt < embedMaxAttempts; attempt++ {
		if err := s.backend.EnsureStarted(); err != nil {
			return nil, fmt.Errorf("backend not ready: %w", err)
		}
		vec, err := s.backend.Embed(ctx, req.Text, true)
		if err == nil {
			return &runedv1.EmbedResponse{Vector: vec}, nil
		}
		if !errors.Is(err, backend.ErrNotStarted) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("backend kept suspending between EnsureStarted and Embed")
}

// EmbedBatch delegates to the backend's batch path and wraps each vector in
// an EmbedResponse so the proto response message stays composable with
// single-text Embed. See Embed godoc on EnsureStarted / cold-start behaviour
// and on the ErrNotStarted retry loop.
func (s *Server) EmbedBatch(ctx context.Context, req *runedv1.EmbedBatchRequest) (*runedv1.EmbedBatchResponse, error) {
	s.requests.Add(1)
	for attempt := 0; attempt < embedMaxAttempts; attempt++ {
		if err := s.backend.EnsureStarted(); err != nil {
			return nil, fmt.Errorf("backend not ready: %w", err)
		}
		vecs, err := s.backend.EmbedBatch(ctx, req.Texts, true)
		if err == nil {
			out := &runedv1.EmbedBatchResponse{
				Embeddings: make([]*runedv1.EmbedResponse, len(vecs)),
			}
			for i, v := range vecs {
				out.Embeddings[i] = &runedv1.EmbedResponse{Vector: v}
			}
			return out, nil
		}
		if !errors.Is(err, backend.ErrNotStarted) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("backend kept suspending between EnsureStarted and EmbedBatch")
}

// Info returns static daemon metadata. Does not touch the backend — safe to
// call before Start() or during a DEGRADED state.
func (s *Server) Info(ctx context.Context, _ *runedv1.InfoRequest) (*runedv1.InfoResponse, error) {
	return &runedv1.InfoResponse{
		DaemonVersion: s.version,
		ModelIdentity: s.modelIdentity,
		VectorDim:     vectorDim,
		MaxTextLength: s.maxTextLength,
		MaxBatchSize:  maxBatchSize,
	}, nil
}

// Health maps backend readiness onto the proto Status enum. A nil backend or
// unhealthy probe yields DEGRADED; we never return an error from this RPC so
// clients can always read uptime as a liveness signal.
func (s *Server) Health(ctx context.Context, _ *runedv1.HealthRequest) (*runedv1.HealthResponse, error) {
	status := runedv1.HealthResponse_STATUS_OK
	if s.backend == nil || !s.backend.IsHealthy(ctx) {
		status = runedv1.HealthResponse_STATUS_DEGRADED
	}
	return &runedv1.HealthResponse{
		Status:        status,
		UptimeSeconds: int64(time.Since(s.startedAt).Seconds()),
		TotalRequests: s.requests.Load(),
	}, nil
}

// Shutdown signals the daemon to begin graceful termination. It closes
// shutdownCh once (guarded by sync.Once); cmd/runed main() observes the
// close and drives GracefulStop + backend.Stop. The RPC itself returns
// immediately — actual drain happens out-of-band.
func (s *Server) Shutdown(ctx context.Context, _ *runedv1.ShutdownRequest) (*runedv1.ShutdownResponse, error) {
	s.shutdownOnce.Do(func() { close(s.shutdownCh) })
	return &runedv1.ShutdownResponse{Accepted: true}, nil
}
