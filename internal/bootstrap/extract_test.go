package bootstrap

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

// buildTarGz writes a gzipped tarball with the given regular-file entries
// and returns its path. Entries are written in map-iteration order, which
// is fine for these tests because the assertions check membership, not
// ordering.
func buildTarGz(t *testing.T, entries map[string][]byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.tar.gz")
	out, err := os.Create(path)
	if err != nil {
		t.Fatalf("create tarball: %v", err)
	}
	defer out.Close()
	gz := gzip.NewWriter(out)
	tw := tar.NewWriter(gz)
	for name, body := range entries {
		if err := tw.WriteHeader(&tar.Header{
			Name:     name,
			Mode:     0o644,
			Size:     int64(len(body)),
			Typeflag: tar.TypeReg,
		}); err != nil {
			t.Fatalf("write tar header for %q: %v", name, err)
		}
		if _, err := tw.Write(body); err != nil {
			t.Fatalf("write tar body for %q: %v", name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	return path
}

// Pins the reviewer's main point: the old strings.Contains(name, "..")
// prefilter dropped legitimate filenames containing literal dots.
// filepath.IsLocal correctly accepts them.
func TestExtractTarGz_AllowsDotsInFilename(t *testing.T) {
	src := buildTarGz(t, map[string][]byte{
		"foo..bar":     []byte("ok"),
		"normal.txt":   []byte("hi"),
		"a.b.c.d.e.f":  []byte("multi"),
		"foo...bar...": []byte("trailing"),
	})
	dst := filepath.Join(t.TempDir(), "out")

	extracted, err := ExtractTarGz(src, dst)
	if err != nil {
		t.Fatalf("ExtractTarGz: %v", err)
	}
	if len(extracted) != 4 {
		t.Errorf("expected 4 extracted files, got %d: %v", len(extracted), extracted)
	}
	for _, name := range []string{"foo..bar", "normal.txt", "a.b.c.d.e.f", "foo...bar..."} {
		if _, err := os.Stat(filepath.Join(dst, name)); err != nil {
			t.Errorf("%q should be extracted: %v", name, err)
		}
	}
}

// Pins the security side: entries that escape the destination via leading
// ".." or absolute paths are silently dropped and never land outside dst.
func TestExtractTarGz_RejectsPathTraversal(t *testing.T) {
	src := buildTarGz(t, map[string][]byte{
		"../escape":     []byte("evil-rel"),
		"../../escape":  []byte("evil-deep"),
		"/etc/absolute": []byte("evil-abs"),
		"safe/legit":    []byte("good"),
	})
	dst := filepath.Join(t.TempDir(), "out")

	extracted, err := ExtractTarGz(src, dst)
	if err != nil {
		t.Fatalf("ExtractTarGz: %v", err)
	}
	if len(extracted) != 1 {
		t.Errorf("expected only the safe entry, got %d: %v", len(extracted), extracted)
	}
	// Sibling of dst must not have been touched by any traversal attempt.
	parent := filepath.Dir(dst)
	if _, err := os.Stat(filepath.Join(parent, "escape")); !os.IsNotExist(err) {
		t.Errorf("escape should not exist outside dest; stat err: %v", err)
	}
}

// Mid-path ".." segments that lexically normalize to within dst are now
// allowed. Documented behavior change: the previous strings.Contains("..")
// prefilter over-rejected this; filepath.IsLocal treats it as in-tree
// because Clean("nested/../escape") = "escape", which lives inside dst.
// Safe because IsLocal guarantees the final path stays under dst.
func TestExtractTarGz_AllowsMidPathDotDotThatStaysInside(t *testing.T) {
	src := buildTarGz(t, map[string][]byte{
		"nested/../escape": []byte("normalized to escape"),
	})
	dst := filepath.Join(t.TempDir(), "out")

	extracted, err := ExtractTarGz(src, dst)
	if err != nil {
		t.Fatalf("ExtractTarGz: %v", err)
	}
	if len(extracted) != 1 {
		t.Errorf("expected 1 extracted file, got %d: %v", len(extracted), extracted)
	}
	if _, err := os.Stat(filepath.Join(dst, "escape")); err != nil {
		t.Errorf("escape should exist inside dst: %v", err)
	}
}
