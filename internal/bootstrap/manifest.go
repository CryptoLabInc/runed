package bootstrap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const ManifestVersion = 1
const manifestFetchTimeout = 30 * time.Second
const manifestMaxBytes = 1 << 20

// Manifest is the subset of the shared rune/runed manifest that runed
// reads. The full manifest may carry extra fields owned by rune install
// (runed, rune_mcp, runed_version, ...); we intentionally do NOT use
// DisallowUnknownFields so this stays forward-compatible.
//
// Example JSON:
//
//	{
//	  "version": 1,
//	  "platforms": {
//	    "darwin-arm64": {
//	      "llama_server": {
//	        "url":     "https://.../llama-bin-macos-arm64.tar.gz",
//	        "sha256":  "...",
//	        "size":    12345678,
//	        "extract": "tar.gz",
//	        "exec":    "build/bin/llama-server"
//	      }
//	    }
//	  },
//	  "models": {
//	    "qwen3-embedding-0.6b.q6_K": {
//	      "url":    "https://huggingface.co/.../qwen3-embedding-0.6b-q6_k.gguf",
//	      "sha256": "...",
//	      "size":   472000000
//	    }
//	  },
//	  "default_model": "qwen3-embedding-0.6b.q6_K"
//	}
type Manifest struct {
	Version      int                          `json:"version"`
	Platforms    map[string]PlatformArtifacts `json:"platforms"`
	Models       map[string]ArtifactSpec      `json:"models"`
	DefaultModel string                       `json:"default_model"`
}

type PlatformArtifacts struct {
	LlamaServer *LlamaServerSpec `json:"llama_server"`
}

// LlamaServerSpec describes how to fetch and unpack the llama-server
// release for a given platform. Extract="" means the URL points at a raw
// executable; Extract="tar.gz" means a gzipped tarball that Exec resolves
// inside.
type LlamaServerSpec struct {
	URL     string `json:"url"`
	SHA256  string `json:"sha256"`
	Size    int64  `json:"size,omitempty"`
	Extract string `json:"extract,omitempty"`
	Exec    string `json:"exec,omitempty"`
}

type ArtifactSpec struct {
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size,omitempty"`
}

var (
	ErrUnsupportedManifestVersion = errors.New("manifest: unsupported version")
	ErrNoArtifactForPlatform      = errors.New("manifest: no llama_server for this platform")
	ErrManifestURLMissing         = errors.New("manifest URL not set; set RUNED_MANIFEST or build with -ldflags '-X .../bootstrap.DefaultManifestURL=...'")
)

// DefaultManifestURL is the manifest URL used when RUNED_MANIFEST is unset.
// Empty in source so dev builds force the env var to be explicit; release
// builds inject the production URL via ldflags so end users don't need to
// know about RUNED_MANIFEST at all.
//
//	go build -ldflags "-X github.com/CryptoLabInc/runed/internal/bootstrap.DefaultManifestURL=https://your.host/manifest.json"
//
// or via the Makefile:
//
//	make build DEFAULT_MANIFEST_URL=https://your.host/manifest.json
var DefaultManifestURL = ""

// ResolveManifestURL returns the URL runed should fetch the manifest from.
// RUNED_MANIFEST env wins over the build-time DefaultManifestURL so
// operators can point at a staging manifest without rebuilding.
func ResolveManifestURL() string {
	if v := os.Getenv(EnvManifest); v != "" {
		return v
	}
	return DefaultManifestURL
}

func FetchManifest(ctx context.Context) (*Manifest, error) {
	url := ResolveManifestURL()
	if url == "" {
		return nil, ErrManifestURLMissing
	}
	return fetchManifestFrom(ctx, url)
}

func fetchManifestFrom(ctx context.Context, url string) (*Manifest, error) {
	fctx, cancel := context.WithTimeout(ctx, manifestFetchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(fctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("manifest: build request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("manifest: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("manifest: GET %s: HTTP %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, manifestMaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("manifest: read body: %w", err)
	}
	if int64(len(body)) > manifestMaxBytes {
		return nil, fmt.Errorf("manifest: body exceeds %d bytes", manifestMaxBytes)
	}
	return parseManifest(body)
}

func parseManifest(body []byte) (*Manifest, error) {
	var m Manifest
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&m); err != nil {
		return nil, fmt.Errorf("manifest: parse JSON: %w", err)
	}
	if m.Version != ManifestVersion {
		return nil, fmt.Errorf("%w: got %d, want %d",
			ErrUnsupportedManifestVersion, m.Version, ManifestVersion)
	}
	return &m, nil
}

func (m *Manifest) LlamaServerForCurrentPlatform() (*LlamaServerSpec, error) {
	tuple := PlatformTuple()
	p, ok := m.Platforms[tuple]
	if !ok || p.LlamaServer == nil {
		return nil, fmt.Errorf("%w: %s", ErrNoArtifactForPlatform, tuple)
	}
	if p.LlamaServer.URL == "" || p.LlamaServer.SHA256 == "" {
		return nil, fmt.Errorf("manifest: llama_server for %s missing url or sha256", tuple)
	}
	if p.LlamaServer.Exec == "" && p.LlamaServer.Extract != "" {
		return nil, fmt.Errorf("manifest: llama_server for %s: extract=%q requires exec",
			tuple, p.LlamaServer.Extract)
	}
	return p.LlamaServer, nil
}

func (m *Manifest) ModelSpec(variant string) (ArtifactSpec, error) {
	if variant == "" {
		return ArtifactSpec{}, errors.New("manifest: model variant not specified")
	}
	spec, ok := m.Models[variant]
	if !ok {
		return ArtifactSpec{}, fmt.Errorf("manifest: model variant %q not in manifest", variant)
	}
	if spec.URL == "" || spec.SHA256 == "" {
		return ArtifactSpec{}, fmt.Errorf("manifest: model %q missing url or sha256", variant)
	}
	return spec, nil
}
