package server

import (
	"errors"
	"io"
	"log/slog"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"github.com/CryptoLabInc/runed/internal/route"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SetCentroidCacheDir sets where a newly pushed centroid set is persisted.
// Called once by cmd/runed before Serve; empty disables persistence.
func (s *Server) SetCentroidCacheDir(dir string) { s.centroidCacheDir = dir }

// LoadCentroids installs a set directly (boot-time cache restore). It
// validates against the daemon's embedding dimension and is a no-op on nil.
func (s *Server) LoadCentroids(cs *route.CentroidSet) error {
	if cs == nil {
		return errors.New("server: nil centroid set")
	}
	if err := cs.Validate(int(vectorDim)); err != nil {
		return err
	}
	s.centroids.Store(cs)
	return nil
}

// SetCentroids assembles the pushed stream (header, then id-ordered batches)
// into a route.CentroidSet, installs it atomically, and persists it to the
// daemon cache best-effort. The push replaces any previous set — the version
// in the header is the engine's content hash, so pushing the same set twice
// is a cheap idempotent overwrite.
func (s *Server) SetCentroids(stream runedv1.RunedService_SetCentroidsServer) error {
	var cs *route.CentroidSet
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		switch p := req.GetPayload().(type) {
		case *runedv1.SetCentroidsRequest_Header:
			if cs != nil {
				return status.Error(codes.InvalidArgument, "duplicate header frame")
			}
			cs = &route.CentroidSet{
				Version: p.Header.GetVersion(),
				Dim:     int(p.Header.GetDim()),
				Preset:  p.Header.GetPreset(),
			}
			if n := p.Header.GetNlist(); n > 0 {
				cs.Vectors = make([][]float32, 0, n)
			}
		case *runedv1.SetCentroidsRequest_Batch:
			if cs == nil {
				return status.Error(codes.InvalidArgument, "batch frame before header")
			}
			for _, c := range p.Batch.GetCentroids() {
				cs.Vectors = append(cs.Vectors, c.GetVec())
			}
		}
	}
	if cs == nil {
		return status.Error(codes.InvalidArgument, "empty stream: header frame required")
	}
	if err := cs.Validate(int(vectorDim)); err != nil {
		return status.Error(codes.InvalidArgument, err.Error())
	}
	// §9.2 C2: the version tag is a content hash — recompute it over the
	// received vectors and reject a push whose content does not match its
	// claim (relay assembly corruption). Skipped when the sender carried no
	// preset (legacy chain), since the hash cannot be recreated without it.
	if err := cs.VerifyVersion(); err != nil {
		return status.Error(codes.InvalidArgument, err.Error())
	}

	s.centroids.Store(cs)
	if s.centroidCacheDir != "" {
		if err := cs.Persist(s.centroidCacheDir); err != nil {
			// Persistence is a restart optimization, not correctness — the set
			// is live in memory; the next push repopulates the cache.
			slog.Warn("centroids: persist failed (set is live in memory)", "err", err, "dir", s.centroidCacheDir)
		}
	}
	slog.Info("centroids: set installed", "version", cs.Version, "nlist", len(cs.Vectors), "dim", cs.Dim)

	return stream.SendAndClose(&runedv1.SetCentroidsResponse{
		Version: cs.Version,
		Nlist:   uint32(len(cs.Vectors)),
	})
}
