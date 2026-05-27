package server

import (
	"context"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"github.com/CryptoLabInc/runed/internal/backend"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// newInProcessServer wires a Server{} to a loopback listener and returns a
// client plus a cleanup that shuts both ends down. It deliberately uses a
// random port (":0") so multiple tests can run in parallel without port clash.
func newInProcessServer(t *testing.T, b *backend.LlamaBackend) (runedv1.RunedServiceClient, func()) {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	runedv1.RegisterRunedServiceServer(gs, New(b, "v0.1.0-test", "test-model-id"))
	go gs.Serve(lis)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	cleanup := func() {
		conn.Close()
		gs.Stop()
		lis.Close()
	}
	return runedv1.NewRunedServiceClient(conn), cleanup
}

// TestServer_EmbedReturnsVector is the end-to-end integration test for the
// Embed RPC. Requires a real llama-server + GGUF — skipped otherwise.
func TestServer_EmbedReturnsVector(t *testing.T) {
	srv := os.Getenv("RUNED_TEST_LLAMA_SERVER")
	gguf := os.Getenv("RUNED_TEST_GGUF")
	if srv == "" || gguf == "" {
		t.Skip("env not set")
	}

	b := backend.NewLlamaBackend(backend.Config{BinaryPath: srv, ModelPath: gguf})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := b.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer b.Stop(context.Background())

	client, cleanup := newInProcessServer(t, b)
	defer cleanup()

	resp, err := client.Embed(ctx, &runedv1.EmbedRequest{Text: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Vector) != 1024 {
		t.Fatalf("want 1024, got %d", len(resp.Vector))
	}
}

// TestServer_InfoReturnsMetadata exercises the Info RPC. It does NOT need a
// running backend because Info only reports static metadata; the nil-backend
// path is intentionally safe.
func TestServer_InfoReturnsMetadata(t *testing.T) {
	b := backend.NewLlamaBackend(backend.Config{}) // not started — Info does not need backend
	client, cleanup := newInProcessServer(t, b)
	defer cleanup()

	info, err := client.Info(context.Background(), &runedv1.InfoRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if info.VectorDim != 1024 {
		t.Fatalf("want dim 1024, got %d", info.VectorDim)
	}
	if info.DaemonVersion != "v0.1.0-test" {
		t.Fatal("version mismatch")
	}
	// Default ctx-size (2048) → advertised max_text_length 2048.
	if info.MaxTextLength != 2048 {
		t.Fatalf("want max_text_length 2048 (default ctx), got %d", info.MaxTextLength)
	}
}

// TestServer_InfoMaxTextLengthTracksCtxSize verifies max_text_length is derived
// from the backend's ctx-size (not a fixed constant), so RUNED_CTX_SIZE tuning
// is reflected to clients through Info. No running backend needed.
func TestServer_InfoMaxTextLengthTracksCtxSize(t *testing.T) {
	b := backend.NewLlamaBackend(backend.Config{CtxSize: 4096}) // not started
	client, cleanup := newInProcessServer(t, b)
	defer cleanup()

	info, err := client.Info(context.Background(), &runedv1.InfoRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if info.MaxTextLength != 4096 {
		t.Fatalf("want max_text_length 4096 (= ctx-size), got %d", info.MaxTextLength)
	}
}

func TestServer_LastActivity_InitializedAtConstruction(t *testing.T) {
	before := time.Now()
	s := New(backend.NewLlamaBackend(backend.Config{}), "vtest", "model-test")
	after := time.Now()
	got := s.LastActivity()
	if got.Before(before) || got.After(after) {
		t.Fatalf("LastActivity = %v; want between %v and %v", got, before, after)
	}
}

func TestServer_TriggerShutdown_Idempotent(t *testing.T) {
	s := New(backend.NewLlamaBackend(backend.Config{}), "vtest", "model-test")
	// Two concurrent TriggerShutdown calls must not panic (sync.Once).
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); s.TriggerShutdown() }()
	}
	wg.Wait()
	// ShutdownCh must be closed exactly once and readable.
	select {
	case _, ok := <-s.ShutdownCh():
		if ok {
			t.Fatal("ShutdownCh should be closed, but got value")
		}
	case <-time.After(time.Second):
		t.Fatal("ShutdownCh not closed after TriggerShutdown")
	}
}

func TestServer_UnaryActivityInterceptor_UpdatesLastActivity(t *testing.T) {
	s := New(backend.NewLlamaBackend(backend.Config{}), "vtest", "model-test")
	initial := s.LastActivity()
	time.Sleep(2 * time.Millisecond) // ensure new nanosecond bucket

	interceptor := s.UnaryActivityInterceptor()
	_, err := interceptor(
		context.Background(),
		struct{}{},
		&grpc.UnaryServerInfo{FullMethod: "/test/Method"},
		func(ctx context.Context, req interface{}) (interface{}, error) {
			return "ok", nil
		},
	)
	if err != nil {
		t.Fatalf("interceptor returned error: %v", err)
	}
	if !s.LastActivity().After(initial) {
		t.Fatalf("LastActivity not advanced: initial=%v after=%v", initial, s.LastActivity())
	}
}

// Before self-bootstrap provide progress, values should be zero
func TestServer_HealthBootstrapFieldsDefaultZero(t *testing.T) {
	b := backend.NewLlamaBackend(backend.Config{}) // not started — Health uses the unhealthy path
	client, cleanup := newInProcessServer(t, b)
	defer cleanup()

	h, err := client.Health(context.Background(), &runedv1.HealthRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if h.Status != runedv1.HealthResponse_STATUS_DEGRADED {
		t.Errorf("Status = %v, want STATUS_DEGRADED (backend not started)", h.Status)
	}
	if h.Phase != runedv1.HealthResponse_PHASE_UNSPECIFIED {
		t.Errorf("Phase = %v, want PHASE_NONE", h.Phase)
	}
	if h.BytesDone != 0 {
		t.Errorf("BytesDone = %d, want 0", h.BytesDone)
	}
	if h.BytesTotal != 0 {
		t.Errorf("BytesTotal = %d, want 0", h.BytesTotal)
	}
	if h.Message != "" {
		t.Errorf("Message = %q, want empty", h.Message)
	}
}
