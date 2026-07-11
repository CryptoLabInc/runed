package route

import (
	"testing"
)

func testSet() *CentroidSet {
	return &CentroidSet{
		Version: "v-abc",
		Dim:     3,
		Vectors: [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
	}
}

func TestValidate(t *testing.T) {
	if err := testSet().Validate(3); err != nil {
		t.Fatalf("valid set rejected: %v", err)
	}
	if err := testSet().Validate(4); err == nil {
		t.Fatal("dim mismatch accepted")
	}
	var nilSet *CentroidSet
	if err := nilSet.Validate(3); err == nil {
		t.Fatal("nil set accepted")
	}
	empty := &CentroidSet{Version: "v", Dim: 3}
	if err := empty.Validate(3); err == nil {
		t.Fatal("empty vectors accepted")
	}
	ragged := testSet()
	ragged.Vectors[1] = []float32{1}
	if err := ragged.Validate(3); err == nil {
		t.Fatal("ragged centroid accepted")
	}
}

func TestAssign(t *testing.T) {
	s := testSet()
	cases := []struct {
		vec  []float32
		want uint32
	}{
		{[]float32{0.9, 0.1, 0}, 0},
		{[]float32{0.1, 0.9, 0}, 1},
		{[]float32{0, 0.2, 0.8}, 2},
		// Tie between 0 and 1 resolves to the lowest id.
		{[]float32{0.5, 0.5, 0}, 0},
	}
	for _, tc := range cases {
		if got := s.Assign(tc.vec); got != tc.want {
			t.Fatalf("Assign(%v) = %d, want %d", tc.vec, got, tc.want)
		}
	}
}

func TestPersistLoadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	orig := testSet()
	if err := orig.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	got, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Version != orig.Version || got.Dim != orig.Dim || len(got.Vectors) != len(orig.Vectors) {
		t.Fatalf("roundtrip mismatch: %+v vs %+v", got, orig)
	}
	if got.Assign([]float32{0, 1, 0}) != 1 {
		t.Fatal("loaded set assigns wrong cluster")
	}
}

func TestLoadMissing(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil {
		t.Fatal("missing cache should error")
	}
}
