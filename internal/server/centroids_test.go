package server

import (
	"io"
	"testing"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"github.com/CryptoLabInc/runed/internal/backend"
	"github.com/CryptoLabInc/runed/internal/route"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// fakeSetCentroidsStream feeds canned frames and captures the close response.
type fakeSetCentroidsStream struct {
	grpc.ServerStream // nil — SetCentroids only uses Recv/SendAndClose
	frames            []*runedv1.SetCentroidsRequest
	resp              *runedv1.SetCentroidsResponse
}

func (f *fakeSetCentroidsStream) Recv() (*runedv1.SetCentroidsRequest, error) {
	if len(f.frames) == 0 {
		return nil, io.EOF
	}
	fr := f.frames[0]
	f.frames = f.frames[1:]
	return fr, nil
}

func (f *fakeSetCentroidsStream) SendAndClose(r *runedv1.SetCentroidsResponse) error {
	f.resp = r
	return nil
}

func header(version string, dim, nlist uint32) *runedv1.SetCentroidsRequest {
	return &runedv1.SetCentroidsRequest{Payload: &runedv1.SetCentroidsRequest_Header{
		Header: &runedv1.CentroidSetHeader{Version: version, Dim: dim, Nlist: nlist},
	}}
}

// headerPreset is header with a non-empty Preset, so the receiver recomputes
// and enforces the content hash instead of skipping it as a legacy push.
func headerPreset(version, preset string, dim, nlist uint32) *runedv1.SetCentroidsRequest {
	return &runedv1.SetCentroidsRequest{Payload: &runedv1.SetCentroidsRequest_Header{
		Header: &runedv1.CentroidSetHeader{Version: version, Preset: preset, Dim: dim, Nlist: nlist},
	}}
}

func batch(vecs ...[]float32) *runedv1.SetCentroidsRequest {
	cs := make([]*runedv1.Centroid, len(vecs))
	for i, v := range vecs {
		cs[i] = &runedv1.Centroid{Id: uint32(i), Vec: v}
	}
	return &runedv1.SetCentroidsRequest{Payload: &runedv1.SetCentroidsRequest_Batch{
		Batch: &runedv1.CentroidBatch{Centroids: cs},
	}}
}

// dimVec returns a vectorDim-length vector with a single 1 at hot.
func dimVec(hot int) []float32 {
	v := make([]float32, vectorDim)
	v[hot] = 1
	return v
}

// TestSetCentroidsInstallsAndPersists drives a well-formed header+batch stream
// and asserts the set is installed in memory (centroids.Load), the close
// response echoes version and nlist, and a reloadable copy lands in the cache.
func TestSetCentroidsInstallsAndPersists(t *testing.T) {
	s := New("test")
	s.SetCentroidCacheDir(t.TempDir())

	st := &fakeSetCentroidsStream{frames: []*runedv1.SetCentroidsRequest{
		header("v1", uint32(vectorDim), 2),
		batch(dimVec(0), dimVec(1)),
	}}
	if err := s.SetCentroids(st); err != nil {
		t.Fatalf("SetCentroids: %v", err)
	}
	if st.resp == nil || st.resp.Version != "v1" || st.resp.Nlist != 2 {
		t.Fatalf("bad response: %+v", st.resp)
	}
	cs := s.centroids.Load()
	if cs == nil || cs.Version != "v1" {
		t.Fatal("set not installed")
	}
	// Persisted copy must be loadable and identical in shape.
	loaded, err := route.Load(s.centroidCacheDir)
	if err != nil {
		t.Fatalf("cache not persisted: %v", err)
	}
	if loaded.Version != "v1" || len(loaded.Vectors) != 2 {
		t.Fatalf("cache mismatch: %+v", loaded)
	}
}

// TestSetCentroidsRejectsBadStreams asserts each malformed stream is rejected
// with InvalidArgument and installs no set. Cases: empty stream, batch before
// header, duplicate header, dim mismatch against the served dimension, empty
// version, and a content-hash mismatch (Preset present so the hash is enforced,
// version tag deliberately wrong) — the last exercises the SetCentroids
// VerifyVersion gate that the no-preset cases skip.
func TestSetCentroidsRejectsBadStreams(t *testing.T) {
	cases := []struct {
		name   string
		frames []*runedv1.SetCentroidsRequest
	}{
		{"empty stream", nil},
		{"batch before header", []*runedv1.SetCentroidsRequest{batch(dimVec(0))}},
		{"duplicate header", []*runedv1.SetCentroidsRequest{
			header("v1", uint32(vectorDim), 1), header("v2", uint32(vectorDim), 1),
		}},
		{"dim mismatch", []*runedv1.SetCentroidsRequest{
			header("v1", 8, 1), batch([]float32{1, 0, 0, 0, 0, 0, 0, 0}),
		}},
		{"empty version", []*runedv1.SetCentroidsRequest{
			header("", uint32(vectorDim), 1), batch(dimVec(0)),
		}},
		{"content hash mismatch", []*runedv1.SetCentroidsRequest{
			headerPreset("sha256:deadbeef", "IP1", uint32(vectorDim), 1), batch(dimVec(0)),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := New("test")
			err := s.SetCentroids(&fakeSetCentroidsStream{frames: tc.frames})
			if status.Code(err) != codes.InvalidArgument {
				t.Fatalf("want InvalidArgument, got %v", err)
			}
			if s.centroids.Load() != nil {
				t.Fatal("bad stream must not install a set")
			}
		})
	}
}

// TestEmbedWithRouteRequiresCentroids checks the routing precondition in
// isolation: with a backend wired (so the bootstrap gate passes) but no
// centroid set, an Embed with_route fails FAILED_PRECONDITION carrying the
// NO_CENTROID_SET reason. The no-centroid check runs before the forward pass,
// so the unstarted backend is never invoked. Asserting the reason (not just the
// code) is what pins this to the routing branch rather than the bootstrap one.
func TestEmbedWithRouteRequiresCentroids(t *testing.T) {
	s := New("test")
	s.SetBackend(backend.NewLlamaBackend(backend.Config{}), "test")

	_, err := s.Embed(t.Context(), &runedv1.EmbedRequest{Text: "hi", WithRoute: true})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("want FailedPrecondition, got %v", err)
	}
	if got := reasonOf(err); got != ReasonNoCentroidSet {
		t.Fatalf("reason = %q, want %q", got, ReasonNoCentroidSet)
	}
}

// TestLoadCentroidsValidates checks the boot-time restore path: LoadCentroids
// rejects a set whose dim differs from the served dimension and accepts a
// matching one, after which Info exposes the loaded centroid version.
func TestLoadCentroidsValidates(t *testing.T) {
	s := New("test")
	bad := &route.CentroidSet{Version: "v", Dim: 8, Vectors: [][]float32{make([]float32, 8)}}
	if err := s.LoadCentroids(bad); err == nil {
		t.Fatal("dim-mismatched cache accepted")
	}
	good := &route.CentroidSet{Version: "v", Dim: int(vectorDim), Vectors: [][]float32{dimVec(3)}}
	if err := s.LoadCentroids(good); err != nil {
		t.Fatalf("valid cache rejected: %v", err)
	}
	info, _ := s.Info(t.Context(), &runedv1.InfoRequest{})
	if info.CentroidSetVersion != "v" {
		t.Fatalf("Info should expose centroid version, got %q", info.CentroidSetVersion)
	}
}
