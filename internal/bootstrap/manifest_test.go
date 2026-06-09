package bootstrap

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fastRetry compresses the retry backoff so tests that exercise the retry
// path don't sleep for real seconds. It restores the original on cleanup.
func fastRetry(t *testing.T) {
	t.Helper()
	orig := downloadRetryBackoff
	downloadRetryBackoff = time.Millisecond
	t.Cleanup(func() { downloadRetryBackoff = orig })
}

const sampleManifest = `{
  "version": 1,
  "runed_version": "v9.9.9",
  "rune_mcp_version": "v9.9.9",
  "platforms": {
    "darwin-arm64": {
      "runed":    {"url": "ignored", "sha256": "ignored"},
      "rune_mcp": {"url": "ignored", "sha256": "ignored"},
      "llama_server": {
        "url": "https://example.com/llama.tar.gz",
        "sha256": "deadbeef",
        "size": 12345,
        "extract": "tar.gz",
        "exec": "build/bin/llama-server"
      }
    },
    "linux-amd64": {
      "llama_server": {
        "url": "https://example.com/llama-linux.tar.gz",
        "sha256": "cafebabe",
        "extract": "tar.gz",
        "exec": "build/bin/llama-server"
      }
    }
  },
  "models": {
    "qwen3-embedding-0.6b.q6_K": {
      "url": "https://example.com/q6k.gguf",
      "sha256": "abc123",
      "size": 472000000
    }
  },
  "default_model": "qwen3-embedding-0.6b.q6_K"
}`

func TestParseManifest_TolerantOfRuneFields(t *testing.T) {
	m, err := parseManifest([]byte(sampleManifest))
	if err != nil {
		t.Fatalf("parseManifest: %v", err)
	}
	if m.Version != 1 {
		t.Errorf("Version: got %d", m.Version)
	}
	if m.DefaultModel != "qwen3-embedding-0.6b.q6_K" {
		t.Errorf("DefaultModel: got %q", m.DefaultModel)
	}
	if got := len(m.Platforms); got != 2 {
		t.Errorf("Platforms count: got %d", got)
	}
	if got := len(m.Models); got != 1 {
		t.Errorf("Models count: got %d", got)
	}
}

func TestParseManifest_VersionMismatch(t *testing.T) {
	_, err := parseManifest([]byte(`{"version": 99}`))
	if !errors.Is(err, ErrUnsupportedManifestVersion) {
		t.Errorf("got %v, want ErrUnsupportedManifestVersion", err)
	}
}

func TestParseManifest_RejectsTrailingData(t *testing.T) {
	body := []byte(sampleManifest + `<garbage>`)
	if _, err := parseManifest(body); err == nil {
		t.Error("expected parse error for trailing data, got nil")
	}
}

func TestModelSpec(t *testing.T) {
	m, _ := parseManifest([]byte(sampleManifest))
	spec, err := m.ModelSpec("qwen3-embedding-0.6b.q6_K")
	if err != nil {
		t.Fatalf("ModelSpec: %v", err)
	}
	if spec.URL != "https://example.com/q6k.gguf" || spec.SHA256 != "abc123" {
		t.Errorf("ModelSpec content unexpected: %+v", spec)
	}

	if _, err := m.ModelSpec("nope"); err == nil {
		t.Error("expected error for unknown variant")
	}
	if _, err := m.ModelSpec(""); err == nil {
		t.Error("expected error for empty variant")
	}
}

func TestResolveManifestURL_EnvWins(t *testing.T) {
	t.Setenv(EnvManifest, "https://runed.example/manifest.json")
	prev := DefaultManifestURL
	DefaultManifestURL = "https://built-in.example/manifest.json"
	t.Cleanup(func() { DefaultManifestURL = prev })

	if got := ResolveManifestURL(); got != "https://runed.example/manifest.json" {
		t.Errorf("got %q, want env value", got)
	}
}

func TestResolveManifestURL_FallsBackToDefault(t *testing.T) {
	t.Setenv(EnvManifest, "")
	prev := DefaultManifestURL
	DefaultManifestURL = "https://built-in.example/manifest.json"
	t.Cleanup(func() { DefaultManifestURL = prev })

	if got := ResolveManifestURL(); got != "https://built-in.example/manifest.json" {
		t.Errorf("got %q, want default", got)
	}
}

func TestResolveManifestURL_BothUnset(t *testing.T) {
	t.Setenv(EnvManifest, "")
	prev := DefaultManifestURL
	DefaultManifestURL = ""
	t.Cleanup(func() { DefaultManifestURL = prev })

	if got := ResolveManifestURL(); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

func TestFetchManifest_HTTP(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(sampleManifest))
	}))
	defer srv.Close()
	t.Setenv(EnvManifest, srv.URL)

	m, err := FetchManifest(t.Context())
	if err != nil {
		t.Fatalf("FetchManifest: %v", err)
	}
	if m.DefaultModel != "qwen3-embedding-0.6b.q6_K" {
		t.Errorf("DefaultModel: got %q", m.DefaultModel)
	}
}

func TestFetchManifest_HTTPError(t *testing.T) {
	fastRetry(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	t.Setenv(EnvManifest, srv.URL)

	_, err := FetchManifest(t.Context())
	if err == nil || !strings.Contains(err.Error(), "500") {
		t.Errorf("expected HTTP 500 error, got %v", err)
	}
}

// TestFetchManifest_RetriesTransient verifies a transient 5xx (e.g. a
// GitHub CDN 504) is retried and the fetch ultimately succeeds.
func TestFetchManifest_RetriesTransient(t *testing.T) {
	fastRetry(t)
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			http.Error(w, "gateway timeout", http.StatusGatewayTimeout)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(sampleManifest))
	}))
	defer srv.Close()
	t.Setenv(EnvManifest, srv.URL)

	m, err := FetchManifest(t.Context())
	if err != nil {
		t.Fatalf("FetchManifest after transient 504s: %v", err)
	}
	if m.Version != 1 {
		t.Errorf("Version = %d, want 1", m.Version)
	}
	if attempts != 3 {
		t.Errorf("attempts = %d, want 3 (two 504s then success)", attempts)
	}
}
