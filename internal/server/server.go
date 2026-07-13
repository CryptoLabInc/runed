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
	"github.com/CryptoLabInc/runed/internal/route"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Machine-readable reasons attached (as an ErrorInfo detail, same pattern as
// runespace's grpcerr) to the two FAILED_PRECONDITION conditions Embed can
// return. The status code stays FAILED_PRECONDITION for both — deliberately
// not retryable at the transport layer (see TestServer_EmbedFailsBeforeBackendSet)
// — but the reasons need opposite application-level handling, so clients
// (rune-mcp) branch on the reason instead of parsing the human message:
//
//	ReasonBootstrapping — model still loading; wait and retry the same call.
//	ReasonNoCentroidSet — push a set via SetCentroids, then retry (§9.2 C4).
const (
	errDomain           = "runed.v1"
	ReasonBootstrapping = "BOOTSTRAPPING"
	ReasonNoCentroidSet = "NO_CENTROID_SET"
)

// preconditionErr builds a FAILED_PRECONDITION status tagged with reason.
// Detail attachment is best-effort: on failure the bare status still carries
// the right code and message.
func preconditionErr(reason, msg string) error {
	st := status.New(codes.FailedPrecondition, msg)
	if d, err := st.WithDetails(&errdetails.ErrorInfo{Reason: reason, Domain: errDomain}); err == nil {
		return d.Err()
	}
	return st.Err()
}

// Plan A constants for the Info RPC. Qwen3-Embedding-0.6B fixes these at
// model-load time; future revisions will source them from config or the
// model file. Hardcoded here because Plan A ships with exactly one model.
const (
	vectorDim    int32 = 1024
	maxBatchSize int32 = 32
)

// bootstrapState is replaced atomically so Health sees a consistent
// {phase, bytes, message} tuple in a single load.
type bootstrapState struct {
	phase      runedv1.HealthResponse_Phase
	bytesDone  int64
	bytesTotal int64
	message    string
}

// Server implements runedv1.RunedServiceServer. Callers (cmd/runed)
// construct New(), then SetBackend once llama-server is up; Health
// reports STATUS_LOADING in between.
type Server struct {
	runedv1.UnimplementedRunedServiceServer

	// nil until SetBackend; while nil Embed returns FAILED_PRECONDITION
	// and Health returns STATUS_LOADING.
	backend atomic.Pointer[backend.LlamaBackend]

	version   string
	startedAt time.Time

	// "" until SetBackend; atomic so concurrent Info reads don't race.
	modelIdentity atomic.Value // string

	// chars equal to backend ctx-size in tokens — chars/token ratio is
	// ≥~1.27 so chars==tokens always fits ctx. 0 before SetBackend.
	maxTextLength atomic.Int32

	// nil before any SetBootstrapStatus call.
	bootstrapStatus atomic.Pointer[bootstrapState]

	// centroids is the IVF set pushed via SetCentroids (or loaded from the
	// disk cache at boot); nil until then, making with_route requests fail
	// with FAILED_PRECONDITION. centroidCacheDir is where a newly pushed set
	// is persisted (empty = no persistence, used by tests).
	centroids        atomic.Pointer[route.CentroidSet]
	centroidCacheDir string

	// Embed + EmbedBatch counter. Surfaced via HealthResponse.total_requests.
	requests atomic.Int64

	// sync.Once guards close(shutdownCh) against double-close.
	shutdownOnce sync.Once
	shutdownCh   chan struct{}

	// UnixNano of the most recent RPC; read by cmd/runed's idle ticker.
	lastActivity atomic.Int64
}

// New returns a Server with backend unset; SetBackend wires it after
// bootstrap completes.
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

// SetBackend wires the backend after bootstrap. maxTextLength and
// modelIdentity are written before backend.Store so any reader seeing
// the backend pointer necessarily sees the other two.
func (s *Server) SetBackend(b *backend.LlamaBackend, modelIdentity string) {
	s.maxTextLength.Store(int32(b.CtxSize()))
	s.modelIdentity.Store(modelIdentity)
	s.backend.Store(b)
}

// SetBootstrapStatus updates the Phase / bytes / message that the next
// Health call returns under STATUS_LOADING. bytesTotal == 0 means the
// total size isn't yet known (no Content-Length observed yet).
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

// Embed returns FAILED_PRECONDITION (not Unavailable) before SetBackend
// so client retry policies don't burn budget against a multi-minute
// bootstrap.
//
// After wiring, EnsureStarted handles idle-suspend wake-up. The retry
// loop covers the EnsureStarted-return → RLock-acquire race where Stop
// slips in and surfaces ErrNotStarted.
func (s *Server) Embed(ctx context.Context, req *runedv1.EmbedRequest) (*runedv1.EmbedResponse, error) {
	b := s.backend.Load()
	if b == nil {
		return nil, preconditionErr(ReasonBootstrapping, "daemon is bootstrapping; embed not yet available")
	}
	// Fail before the forward pass: routing without a centroid set can never
	// succeed, and the caller (rune-mcp) reacts by pushing SetCentroids first.
	cs := s.centroids.Load()
	if req.WithRoute && cs == nil {
		return nil, preconditionErr(ReasonNoCentroidSet, "no centroid set loaded; push one via SetCentroids before requesting with_route")
	}
	s.requests.Add(1)
	for attempt := 0; attempt < embedMaxAttempts; attempt++ {
		if err := b.EnsureStarted(); err != nil {
			return nil, fmt.Errorf("backend not ready: %w", err)
		}
		vec, err := b.Embed(ctx, req.Text, true)
		if err == nil {
			resp := &runedv1.EmbedResponse{Vector: vec}
			if req.WithRoute {
				resp.ClusterId = cs.Assign(vec)
				resp.CentroidSetVersion = cs.Version
			}
			return resp, nil
		}
		if !errors.Is(err, backend.ErrNotStarted) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("backend kept suspending between EnsureStarted and Embed")
}

// EmbedBatch is the batch variant of Embed; see Embed for behaviour
// during bootstrap and idle-suspend.
func (s *Server) EmbedBatch(ctx context.Context, req *runedv1.EmbedBatchRequest) (*runedv1.EmbedBatchResponse, error) {
	b := s.backend.Load()
	if b == nil {
		return nil, preconditionErr(ReasonBootstrapping, "daemon is bootstrapping; embed not yet available")
	}
	cs := s.centroids.Load()
	if req.WithRoute && cs == nil {
		return nil, preconditionErr(ReasonNoCentroidSet, "no centroid set loaded; push one via SetCentroids before requesting with_route")
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
				if req.WithRoute {
					out.Embeddings[i].ClusterId = cs.Assign(v)
					out.Embeddings[i].CentroidSetVersion = cs.Version
				}
			}
			return out, nil
		}
		if !errors.Is(err, backend.ErrNotStarted) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("backend kept suspending between EnsureStarted and EmbedBatch")
}

// Info returns static daemon metadata. MaxTextLength reads 0 before
// SetBackend; clients needing the final value should re-query after
// Health reports STATUS_OK.
func (s *Server) Info(ctx context.Context, _ *runedv1.InfoRequest) (*runedv1.InfoResponse, error) {
	mid, _ := s.modelIdentity.Load().(string)
	info := &runedv1.InfoResponse{
		DaemonVersion: s.version,
		ModelIdentity: mid,
		VectorDim:     vectorDim,
		MaxTextLength: s.maxTextLength.Load(),
		MaxBatchSize:  maxBatchSize,
	}
	if cs := s.centroids.Load(); cs != nil {
		info.CentroidSetVersion = cs.Version
	}
	return info, nil
}

// Health maps backend state to the proto Status enum. SHUTTING_DOWN
// outranks LOADING/IDLE/DEGRADED/OK so a drain-in-progress daemon doesn't
// advertise itself as ready (the GracefulStop race would otherwise
// surface Unavailable right after the OK response).
//
// Never returns an error — clients can read uptime as a liveness signal
// even when the other fields are zero-valued.
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
	switch b.Serving(ctx) {
	case backend.ServingIdle:
		resp.Status = runedv1.HealthResponse_STATUS_IDLE
		resp.Message = "embedder suspended after idle to free memory; the next request resumes it automatically"
	case backend.ServingDegraded:
		resp.Status = runedv1.HealthResponse_STATUS_DEGRADED
	default:
		resp.Status = runedv1.HealthResponse_STATUS_OK
	}
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
