package route

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func verifiedSet() *CentroidSet {
	s := &CentroidSet{
		Dim:    2,
		Preset: "IP1",
		Vectors: [][]float32{
			{1, 0},
			{0.5, 0.5},
		},
	}
	s.Version = ComputeVersion(s)
	return s
}

// ComputeVersion must follow the engine's exact recipe: sha256 over dim(LE),
// nlist(LE), preset, NUL, then float32 bits in id order (LE). This golden
// test hand-assembles those bytes independently so a silent recipe drift in
// either implementation breaks it.
func TestComputeVersionRecipe(t *testing.T) {
	s := verifiedSet()
	var buf []byte
	u32 := func(v uint32) { b := make([]byte, 4); binary.LittleEndian.PutUint32(b, v); buf = append(buf, b...) }
	u32(2) // dim
	u32(2) // nlist
	buf = append(buf, []byte("IP1")...)
	buf = append(buf, 0)
	for _, vec := range s.Vectors {
		for _, f := range vec {
			u32(math.Float32bits(f))
		}
	}
	// 골든 비교: 손으로 조립한 재료 바이트를 직접 해싱해, 구현이 dim·nlist 순서,
	// 엔디안, preset NUL 구분자까지 바이트 단위로 동일한 레시피를 따르는지 고정한다.
	sum := sha256.Sum256(buf)
	want := "sha256:" + hex.EncodeToString(sum[:])
	got := ComputeVersion(s)
	if got != want {
		t.Fatalf("recipe drift: ComputeVersion=%s, hand-assembled=%s", got, want)
	}
	if !strings.HasPrefix(got, "sha256:") {
		t.Fatalf("version prefix missing: %s", got)
	}
	// 같은 재료로 두 번 계산하면 결정적이어야 하고, 재료 하나만 바뀌어도 달라져야 한다.
	if ComputeVersion(s) != s.Version {
		t.Fatal("ComputeVersion is not deterministic")
	}
	preset2 := *s
	preset2.Preset = "IP2"
	if ComputeVersion(&preset2) == s.Version {
		t.Fatal("preset must be a hash ingredient")
	}
}

// TestVerifyVersion exercises the integrity gate: a set whose content matches
// its tag passes, content mutated under an intact tag is rejected, and a legacy
// set carrying no Preset skips verification (the hash can't be recomputed).
func TestVerifyVersion(t *testing.T) {
	s := verifiedSet()
	if err := s.VerifyVersion(); err != nil {
		t.Fatalf("valid set rejected: %v", err)
	}
	// 내용만 바뀌고 꼬리표는 그대로 — 무증상 손상의 형태 그대로 재현
	s.Vectors[0][0] = 0.9999
	if err := s.VerifyVersion(); err == nil {
		t.Fatal("content corruption with intact tag was not detected")
	}
	// 레거시(전달자에 preset 없음): 재계산 불가 → 스킵
	legacy := &CentroidSet{Version: "sha256:whatever", Dim: 2, Vectors: [][]float32{{1, 0}}}
	if err := legacy.VerifyVersion(); err != nil {
		t.Fatalf("legacy (no preset) must skip verification: %v", err)
	}
}

// C5 end-to-end at the file layer: a bit flipped inside the gob's float
// payload decodes cleanly but must now fail Load via the hash recomputation.
func TestLoadDetectsBitrot(t *testing.T) {
	dir := t.TempDir()
	s := verifiedSet()
	if err := s.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	if _, err := Load(dir); err != nil {
		t.Fatalf("clean roundtrip must verify: %v", err)
	}

	path := filepath.Join(dir, "centroids.gob")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// float32(1.0)의 gob 표현 바이트를 찾아 한 비트 뒤집는다 — 구조는 유지되고
	// 내용만 변하는 손상. (float 위치를 못 찾으면 마지막 바이트를 뒤집는다.)
	flipped := false
	for i := len(raw) - 1; i > 0; i-- {
		cand := append(append([]byte{}, raw[:i]...), raw[i]^0x01)
		cand = append(cand, raw[i+1:]...)
		if err := os.WriteFile(path, cand, 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := Load(dir)
		if err != nil && strings.Contains(err.Error(), "cache verify") {
			flipped = true // 원하는 경로: decode 성공 + verify가 잡음
			break
		}
		if err != nil {
			continue // decode 자체가 깨진 위치 — 다른 바이트로 재시도
		}
		_ = got
		t.Fatalf("corrupted cache loaded without complaint (byte %d)", i)
	}
	if !flipped {
		t.Skip("no byte position produced intact-gob corruption in this encoding")
	}
}
