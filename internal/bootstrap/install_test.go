package bootstrap

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
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

func TestEnsureLlamaServer_PlatformMissingFails(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	p, _ := Resolve()

	m := &Manifest{
		Version:   1,
		Platforms: map[string]PlatformArtifacts{},
		Models:    map[string]ArtifactSpec{},
	}

	if _, err := EnsureLlamaServer(t.Context(), p, m, slog.Default(), nil); !errors.Is(err, ErrNoArtifactForPlatform) {
		t.Fatalf("expected ErrNoArtifactForPlatform, got: %v", err)
	}
}

// A caller on a platform missing from the manifest can still bootstrap
// the model side (LlamaServerForCurrentPlatform is not consulted).
func TestEnsureModel_NoLlamaServerEntryNeeded(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")
	p, _ := Resolve()
	if err := p.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}

	// Pre-populate the model file so the cache-hit path returns without
	// touching the network — what we're testing is the platform-check path,
	// not the download path.
	variant := "test-variant"
	body := []byte("model-content")
	sum := sha256.Sum256(body)
	wantSHA := hex.EncodeToString(sum[:])

	modelPath := p.ModelPath(variant)
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o700); err != nil {
		t.Fatalf("mkdir models: %v", err)
	}
	if err := os.WriteFile(modelPath, body, 0o600); err != nil {
		t.Fatalf("write model: %v", err)
	}

	m := &Manifest{
		Version:   1,
		Platforms: map[string]PlatformArtifacts{}, // intentionally empty
		Models: map[string]ArtifactSpec{
			variant: {URL: "https://ignored", SHA256: wantSHA, Size: int64(len(body))},
		},
		DefaultModel: variant,
	}

	got, gotVariant, err := EnsureModel(t.Context(), p, m, slog.Default(), nil)
	if err != nil {
		t.Fatalf("EnsureModel must succeed when no llama-server entry is present: %v", err)
	}
	if got != modelPath {
		t.Errorf("got path %q, want %q", got, modelPath)
	}
	if gotVariant != variant {
		t.Errorf("got variant %q, want %q", gotVariant, variant)
	}
}

// Stage tick fires on Ensure* entry so Health flips Phase even when
// cache hit (or early error) means no byte ticks follow.
func TestEnsureLlamaServer_ReporterReceivesStageTickOnEntry(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	p, _ := Resolve()

	// platforms empty → LlamaServerForCurrentPlatform errors after the
	// reporter's entry tick.
	m := &Manifest{
		Version:   1,
		Platforms: map[string]PlatformArtifacts{},
		Models:    map[string]ArtifactSpec{},
	}

	var stages []string
	reporter := func(stage string, _, _ int64) {
		stages = append(stages, stage)
	}
	_, _ = EnsureLlamaServer(t.Context(), p, m, slog.Default(), reporter)
	if len(stages) != 1 || stages[0] != "llama_server" {
		t.Errorf("stages = %v, want [llama_server]", stages)
	}
}

func TestEnsureModel_ReporterReceivesStageTickOnEntry(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")
	p, _ := Resolve()
	if err := p.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}

	// Pre-populate model so cache-hit returns without any download path
	// running. Only the entry tick should reach the reporter.
	variant := "test-variant"
	body := []byte("model-content")
	sum := sha256.Sum256(body)
	wantSHA := hex.EncodeToString(sum[:])
	modelPath := p.ModelPath(variant)
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o700); err != nil {
		t.Fatalf("mkdir models: %v", err)
	}
	if err := os.WriteFile(modelPath, body, 0o600); err != nil {
		t.Fatalf("write model: %v", err)
	}

	m := &Manifest{
		Version: 1,
		Models: map[string]ArtifactSpec{
			variant: {URL: "https://ignored", SHA256: wantSHA, Size: int64(len(body))},
		},
		DefaultModel: variant,
	}

	var stages []string
	reporter := func(stage string, _, _ int64) {
		stages = append(stages, stage)
	}
	if _, _, err := EnsureModel(t.Context(), p, m, slog.Default(), reporter); err != nil {
		t.Fatalf("EnsureModel: %v", err)
	}
	if len(stages) != 1 || stages[0] != "model" {
		t.Errorf("stages = %v, want [model]", stages)
	}
}
