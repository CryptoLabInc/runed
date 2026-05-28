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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Plan A constants for the Info RPC. Qwen3-Embedding-0.6B fixes these at
// model-load time; future revisions will source them from config or the
// model file. Hardcoded here because Plan A ships with exactly one model.
const (
	vectorDim    int32 = 1024
	maxBatchSize int32 = 32
)

// bootstrapState is the snapshot fed to Health while STATUS_LOADING.
// Treated as immutable once stored — SetBootstrapStatus replaces the
// whole pointer atomically so readers always see a consistent tuple.
type bootstrapState struct {
	phase      runedv1.HealthResponse_Phase
	bytesDone  int64
	bytesTotal int64
	message    string
}

// Server implements runedv1.RunedServiceServer. It does not own the backend
// lifecycle — callers (cmd/runed) construct New(), drive self-bootstrap while
// the gRPC socket already listens (Health reports STATUS_LOADING during that
// window), then call SetBackend once llama-server is up.
type Server struct {
	runedv1.UnimplementedRunedServiceServer

	// backend is nil until SetBackend is called. Embed/EmbedBatch return
	// FAILED_PRECONDITION while nil; Health returns STATUS_LOADING.
	backend atomic.Pointer[backend.LlamaBackend]

	version   string
	startedAt time.Time

	// modelIdentity is "" until SetBackend computes the model SHA. Stored
	// via atomic.Value so Info readers don't race the writer.
	modelIdentity atomic.Value // string

	// maxTextLength (chars) is sourced from the backend's ctx-size (tokens)
	// at SetBackend time; chars==tokens is conservative (dense text is
	// ≥~1.27 chars/token), so it always fits ctx. Advertised via Info; reads
	// 0 before bootstrap completes since the value depends on llama-server's
	// loaded config.
	maxTextLength atomic.Int32

	// bootstrapStatus is updated during self-bootstrap and read by Health
	// when backend is still nil. nil before any update.
	bootstrapStatus atomic.Pointer[bootstrapState]

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

// New returns a Server with backend unset. Until SetBackend is called,
// Embed/EmbedBatch return FAILED_PRECONDITION and Health reports
// STATUS_LOADING + whatever phase the latest SetBootstrapStatus posted.
// modelIdentity is empty until SetBackend supplies it.
func New(version string) *Server {
	s := &Server{
		version:    version,
		startedAt:  time.Now(),
		shutdownCh: make(chan struct{}),
	}
	s.modelIdentity.Store("")
	s.lastActivity.Store(time.Now().UnixNano())
	return s
}

// SetBackend wires the backend and model identity after self-bootstrap
// completes. From this point on, Embed/EmbedBatch are accepted and
// Health reports STATUS_OK (or STATUS_DEGRADED if IsHealthy fails). Safe
// to call concurrently with in-flight RPCs — readers see a consistent
// transition because maxTextLength and modelIdentity are written before
// the backend pointer is published.
func (s *Server) SetBackend(b *backend.LlamaBackend, modelIdentity string) {
	s.maxTextLength.Store(int32(b.CtxSize()))
	s.modelIdentity.Store(modelIdentity)
	s.backend.Store(b)
}

// SetBootstrapStatus records the current self-bootstrap phase and download
// progress. The next Health RPC returns these fields when STATUS_LOADING.
// Callers (cmd/runed + bootstrap reporter) emit one update per phase
// transition and periodically during long downloads. bytesTotal == 0
// means total size isn't yet known (e.g. before HTTP Content-Length is
// read); clients should render percent-complete only when total > 0.
func (s *Server) SetBootstrapStatus(phase runedv1.HealthResponse_Phase, bytesDone, bytesTotal int64, message string) {
	s.bootstrapStatus.Store(&bootstrapState{
		phase:      phase,
		bytesDone:  bytesDone,
		bytesTotal: bytesTotal,
		message:    message,
	})
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
// Returns FAILED_PRECONDITION when the backend hasn't been wired yet
// (self-bootstrap still in progress). codes.FailedPrecondition (not
// Unavailable) intentionally bypasses default-retry policies — bootstrap
// can take minutes, so short exponential backoffs would just exhaust
// retries pre-ready. Whether clients fast-fail or poll Health is the
// client's concern; the error message stays neutral on retry strategy.
//
// Once the backend is wired it may still be suspended by the idle-
// suspend ticker. EnsureStarted resurrects it under the daemon-lifetime
// context — the first request after a suspend pays llama-server cold-
// start latency (~hundreds of ms to a few seconds for model load);
// subsequent requests fall through the cheap health-probe fast path.
//
// Retry loop: backend.Embed holds inflightMu.RLock so Stop can't kill
// an in-flight HTTP. The remaining race window is EnsureStarted-return
// → RLock-acquire; if Stop slips into that gap we get ErrNotStarted on
// the first attempt and recover by re-running EnsureStarted once.
func (s *Server) Embed(ctx context.Context, req *runedv1.EmbedRequest) (*runedv1.EmbedResponse, error) {
	b := s.backend.Load()
	if b == nil {
		return nil, status.Error(codes.FailedPrecondition, "daemon is bootstrapping; embed not yet available")
	}
	s.requests.Add(1)
	for attempt := 0; attempt < embedMaxAttempts; attempt++ {
		if err := b.EnsureStarted(); err != nil {
			return nil, fmt.Errorf("backend not ready: %w", err)
		}
		vec, err := b.Embed(ctx, req.Text, true)
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
// single-text Embed. Returns FAILED_PRECONDITION when the backend hasn't
// been wired yet; see Embed godoc for the EnsureStarted / ErrNotStarted
// retry rationale.
func (s *Server) EmbedBatch(ctx context.Context, req *runedv1.EmbedBatchRequest) (*runedv1.EmbedBatchResponse, error) {
	b := s.backend.Load()
	if b == nil {
		return nil, status.Error(codes.FailedPrecondition, "daemon is bootstrapping; embed not yet available")
	}
	s.requests.Add(1)
	for attempt := 0; attempt < embedMaxAttempts; attempt++ {
		if err := b.EnsureStarted(); err != nil {
			return nil, fmt.Errorf("backend not ready: %w", err)
		}
		vecs, err := b.EmbedBatch(ctx, req.Texts, true)
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
// call before SetBackend or during a DEGRADED state. MaxTextLength reads 0
// before bootstrap completes since the value depends on llama-server's
// loaded ctx-size; clients should re-query Info after Health reports
// STATUS_OK if they need the final value.
func (s *Server) Info(ctx context.Context, _ *runedv1.InfoRequest) (*runedv1.InfoResponse, error) {
	mid, _ := s.modelIdentity.Load().(string)
	return &runedv1.InfoResponse{
		DaemonVersion: s.version,
		ModelIdentity: mid,
		VectorDim:     vectorDim,
		MaxTextLength: s.maxTextLength.Load(),
		MaxBatchSize:  maxBatchSize,
	}, nil
}

// Health maps backend readiness onto the proto Status enum:
//
//   - shutdown signalled (Shutdown RPC / TriggerShutdown) → STATUS_SHUTTING_DOWN
//   - backend not yet wired                              → STATUS_LOADING +
//     Phase / bytes / message populated from the most recent
//     SetBootstrapStatus
//   - backend wired but unhealthy                         → STATUS_DEGRADED
//   - backend wired and healthy                           → STATUS_OK
//
// SHUTTING_DOWN is checked first so a drain-in-progress daemon doesn't
// advertise itself as ready (callers that read OK during the GracefulStop
// race would otherwise send a request just to receive Unavailable).
//
// Never returns an error so clients can always read uptime as a liveness
// signal and treat any RPC success as proof the daemon process exists.
func (s *Server) Health(ctx context.Context, _ *runedv1.HealthRequest) (*runedv1.HealthResponse, error) {
	resp := &runedv1.HealthResponse{
		UptimeSeconds: int64(time.Since(s.startedAt).Seconds()),
		TotalRequests: s.requests.Load(),
	}
	select {
	case <-s.shutdownCh:
		resp.Status = runedv1.HealthResponse_STATUS_SHUTTING_DOWN
		return resp, nil
	default:
	}
	b := s.backend.Load()
	if b == nil {
		resp.Status = runedv1.HealthResponse_STATUS_LOADING
		if bs := s.bootstrapStatus.Load(); bs != nil {
			resp.Phase = bs.phase
			resp.BytesDone = bs.bytesDone
			resp.BytesTotal = bs.bytesTotal
			resp.Message = bs.message
		}
		return resp, nil
	}
	if !b.IsHealthy(ctx) {
		resp.Status = runedv1.HealthResponse_STATUS_DEGRADED
		return resp, nil
	}
	resp.Status = runedv1.HealthResponse_STATUS_OK
	return resp, nil
}

// Shutdown signals the daemon to begin graceful termination. It closes
// shutdownCh once (guarded by sync.Once); cmd/runed main() observes the
// close and drives GracefulStop + backend.Stop. The RPC itself returns
// immediately — actual drain happens out-of-band.
func (s *Server) Shutdown(ctx context.Context, _ *runedv1.ShutdownRequest) (*runedv1.ShutdownResponse, error) {
	s.shutdownOnce.Do(func() { close(s.shutdownCh) })
	return &runedv1.ShutdownResponse{Accepted: true}, nil
}
