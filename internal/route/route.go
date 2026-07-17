// Package route holds the daemon's IVF centroid set and the plaintext
// cluster assignment used to route inserts. runed never dials the index
// engine: the set arrives via the SetCentroids RPC (relayed
// runespace → console → rune-mcp → runed) and is cached on disk so a restart
// does not need a re-push. Assignment must agree byte-for-byte with the
// engine's insert routing: max inner product over l2-normalized vectors,
// ties to the lowest id (mirrors runespace-go-sdk clustering).
package route

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/gob"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// CentroidSet is an immutable snapshot of the engine's IVF centroid set.
// Vectors[i] is cluster i's centroid (id order); Version is the set's
// content hash and travels with every routed assignment so the engine can
// reject routing done against a stale set. Preset is the evi preset the set
// was trained for — a Version-hash ingredient, so carrying it is what makes
// local recomputation (VerifyVersion) possible.
type CentroidSet struct {
	Version string
	Dim     int
	Preset  string
	Vectors [][]float32
}

// versionPrefix mirrors the engine's content-hash tag.
const versionPrefix = "sha256:"

// ComputeVersion recreates the engine's content hash over the set: sha256 of
// dim, nlist, preset, then every centroid's float32 bits in id order
// (little-endian). MUST stay byte-identical to runespace's computeVersion
// (runespace/internal/cluster/centroid.go) — that equality is what lets runed
// verify a relayed or cached set against the version the engine minted.
func ComputeVersion(s *CentroidSet) string {
	h := sha256.New()
	var u32 [4]byte
	binary.LittleEndian.PutUint32(u32[:], uint32(s.Dim))
	h.Write(u32[:])
	binary.LittleEndian.PutUint32(u32[:], uint32(len(s.Vectors)))
	h.Write(u32[:])
	h.Write([]byte(s.Preset))
	h.Write([]byte{0}) // length-delimit the preset string
	for _, vec := range s.Vectors {
		for _, f := range vec {
			binary.LittleEndian.PutUint32(u32[:], math.Float32bits(f))
			h.Write(u32[:])
		}
	}
	return versionPrefix + hex.EncodeToString(h.Sum(nil))
}

// VerifyVersion recomputes the content hash and compares it to the carried
// Version — the integrity gate at the set's two entry points (SetCentroids
// receipt, disk-cache load). An empty Preset means the sender predates the
// preset relay: the hash cannot be recreated without that ingredient, so
// verification is skipped (legacy behavior, trust the tag).
func (s *CentroidSet) VerifyVersion() error {
	if s.Preset == "" {
		return nil
	}
	if got := ComputeVersion(s); got != s.Version {
		return fmt.Errorf("route: centroid content does not match its version tag (tag %s, content hashes to %s) — corrupt relay or cache", s.Version, got)
	}
	return nil
}

// Validate reports whether the set is internally consistent and matches the
// embedding dimension the daemon serves.
func (s *CentroidSet) Validate(wantDim int) error {
	if s == nil || s.Version == "" || len(s.Vectors) == 0 {
		return errors.New("route: centroid set is empty")
	}
	if s.Dim != wantDim {
		return fmt.Errorf("route: centroid dim %d does not match embedding dim %d", s.Dim, wantDim)
	}
	for i, v := range s.Vectors {
		if len(v) != s.Dim {
			return fmt.Errorf("route: centroid %d has dim %d, want %d", i, len(v), s.Dim)
		}
	}
	return nil
}

// Assign returns the cluster id whose centroid has the highest inner product
// with vec. vec must be l2-normalized — runed embeddings already are
// (llama.cpp last-pooling normalizes unconditionally). Ties resolve to the
// lowest id, matching the SDK/engine metric exactly.
func (s *CentroidSet) Assign(vec []float32) uint32 {
	bestID := uint32(0)
	best := math.Inf(-1)
	for i, c := range s.Vectors {
		var dot float64
		for j := 0; j < len(c) && j < len(vec); j++ {
			dot += float64(c[j]) * float64(vec[j])
		}
		if dot > best {
			best, bestID = dot, uint32(i)
		}
	}
	return bestID
}

const cacheFile = "centroids.gob"

// Persist writes the set atomically to dir (temp + rename), so a torn write
// never produces a half-cache. Best-effort semantics are the caller's call.
func (s *CentroidSet) Persist(dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("route: cache dir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, cacheFile+".tmp-*")
	if err != nil {
		return fmt.Errorf("route: cache temp: %w", err)
	}
	defer os.Remove(tmp.Name())
	if err := gob.NewEncoder(tmp).Encode(s); err != nil {
		tmp.Close()
		return fmt.Errorf("route: cache encode: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("route: cache close: %w", err)
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, cacheFile)); err != nil {
		return fmt.Errorf("route: cache rename: %w", err)
	}
	return nil
}

// Load reads a previously persisted set. A missing cache returns
// (nil, fs.ErrNotExist-wrapped error); a corrupt cache returns an error and
// the caller should discard it and wait for the next SetCentroids push.
// Corruption covers both shapes: a broken gob (decode error) and — the
// insidious case — intact gob framing around flipped content bytes, which
// only the §9.2 C5 hash recomputation can see.
func Load(dir string) (*CentroidSet, error) {
	f, err := os.Open(filepath.Join(dir, cacheFile))
	if err != nil {
		return nil, fmt.Errorf("route: cache open: %w", err)
	}
	defer f.Close()
	var s CentroidSet
	if err := gob.NewDecoder(f).Decode(&s); err != nil {
		return nil, fmt.Errorf("route: cache decode: %w", err)
	}
	if err := s.VerifyVersion(); err != nil {
		return nil, fmt.Errorf("route: cache verify: %w", err)
	}
	return &s, nil
}
