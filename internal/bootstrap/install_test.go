package bootstrap

import (
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestResolveModelVariant_EnvWins(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "from-env")

	p, _ := Resolve()
	// Even with config + manifest defaults set, env wins.
	writeConfig(t, p.Config, `{"version":1,"model_variant":"from-config"}`)
	m := &Manifest{DefaultModel: "from-manifest"}

	got, err := ResolveModelVariant(p, m)
	if err != nil {
		t.Fatalf("ResolveModelVariant: %v", err)
	}
	if got != "from-env" {
		t.Errorf("got %q, want from-env", got)
	}
}

func TestResolveModelVariant_ConfigOverManifest(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")

	p, _ := Resolve()
	writeConfig(t, p.Config, `{"version":1,"model_variant":"from-config"}`)
	m := &Manifest{DefaultModel: "from-manifest"}

	got, err := ResolveModelVariant(p, m)
	if err != nil {
		t.Fatalf("ResolveModelVariant: %v", err)
	}
	if got != "from-config" {
		t.Errorf("got %q, want from-config", got)
	}
}

func TestResolveModelVariant_ManifestDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")

	p, _ := Resolve()
	// No config file written.
	m := &Manifest{DefaultModel: "from-manifest"}

	got, err := ResolveModelVariant(p, m)
	if err != nil {
		t.Fatalf("ResolveModelVariant: %v", err)
	}
	if got != "from-manifest" {
		t.Errorf("got %q, want from-manifest", got)
	}
}

func TestResolveModelVariant_NoneSpecified(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")

	p, _ := Resolve()
	m := &Manifest{}
	if _, err := ResolveModelVariant(p, m); err == nil {
		t.Fatal("expected error when no variant source set")
	}
}

func TestLlamaServerTarget_DefaultExec(t *testing.T) {
	p := &Paths{LlamaDir: "/x/bin/llama-cpp"}
	got := llamaServerTarget(p, &LlamaServerSpec{}) // Exec unset
	want := filepath.Join("/x/bin/llama-cpp", "llama-server")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestLlamaServerTarget_NestedExec(t *testing.T) {
	p := &Paths{LlamaDir: "/x/bin/llama-cpp"}
	got := llamaServerTarget(p, &LlamaServerSpec{Exec: "build/bin/llama-server"})
	want := filepath.Join("/x/bin/llama-cpp", "build", "bin", "llama-server")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestDownloadWithRetry_SucceedsAfterTransientFailures(t *testing.T) {
	prev := downloadRetryBackoff
	t.Cleanup(func() { downloadRetryBackoff = prev })
	downloadRetryBackoff = time.Millisecond

	body := []byte("good")
	sum := sha256.Sum256(body)
	wantSHA := hex.EncodeToString(sum[:])

	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n < 3 {
			http.Error(w, "transient", http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := downloadWithRetry(t.Context(), srv.URL, wantSHA, int64(len(body)), dest, nil, slog.Default(), "test"); err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("expected 3 attempts, got %d", got)
	}
}

func TestDownloadWithRetry_GivesUpAfterMaxAttempts(t *testing.T) {
	prev := downloadRetryBackoff
	t.Cleanup(func() { downloadRetryBackoff = prev })
	downloadRetryBackoff = time.Millisecond

	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "always-fails", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	err := downloadWithRetry(t.Context(), srv.URL, "deadbeef", 4, dest, nil, slog.Default(), "test")
	if err == nil {
		t.Fatal("expected failure after exhausting retries")
	}
	if !strings.Contains(err.Error(), "after 3 attempts") {
		t.Errorf("error should mention attempt count; got: %v", err)
	}
	if got := calls.Load(); got != int32(maxDownloadAttempts) {
		t.Errorf("expected %d attempts, got %d", maxDownloadAttempts, got)
	}
}

func writeConfig(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
}
