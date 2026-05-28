package bootstrap

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDownloadAndVerify_RejectsContentLengthMismatch(t *testing.T) {
	body := []byte("short body")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	err := DownloadAndVerify(t.Context(), srv.URL, "deadbeef", 99999, dest, nil)
	if err == nil {
		t.Fatal("expected Content-Length mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "Content-Length mismatch") {
		t.Errorf("expected Content-Length mismatch error, got: %v", err)
	}
	// Fail-fast must skip the .partial creation entirely — otherwise the
	// reviewer's point (don't waste I/O) is only half kept.
	if _, statErr := os.Stat(dest + ".partial"); !os.IsNotExist(statErr) {
		t.Errorf(".partial should not exist; stat err: %v", statErr)
	}
}
