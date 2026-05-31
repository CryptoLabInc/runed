package bootstrap

import (
	"encoding/json"
	"os"
	"testing"
)

func newAuditTestPaths(t *testing.T) *Paths {
	t.Helper()
	t.Setenv(EnvHome, t.TempDir())

	paths, err := Resolve()
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if err := paths.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}

	return paths
}

func TestWriteAndReadInstalledManifest_RoundTrip(t *testing.T) {
	paths := newAuditTestPaths(t)

	manifest := &InstalledManifest{
		ManifestURL:     "https://example/manifest.json",
		ManifestVersion: 1,
		Platform:        PlatformTuple(),
		InstalledAt:     "2026-05-31T00:00:00Z",
		ModelVariant:    "qwen3-embedding-0.6b.q6_K",
		Artifacts: map[string]InstalledArtifact{
			AuditArtifactLlamaServer: {URL: "https://example/llama.tar.gz", SHA256: "aaa", Path: "/x/bin/llama-server", Size: 1234},
			AuditArtifactModel:       {URL: "https://example/qwen3.gguf", SHA256: "bbb", Path: "/x/models/v.gguf", Size: 5678},
		},
	}
	if err := WriteInstalledManifest(paths, manifest); err != nil {
		t.Fatalf("WriteInstalledManifest: %v", err)
	}

	info, err := os.Stat(InstalledManifestPath(paths))
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("mode = %v, want 0o600", perm)
	}

	got, err := ReadInstalledManifest(paths)
	if err != nil {
		t.Fatalf("ReadInstalledManifest: %v", err)
	}
	if got.ManifestURL != manifest.ManifestURL ||
		got.ManifestVersion != manifest.ManifestVersion ||
		got.ModelVariant != manifest.ModelVariant {
		t.Errorf("top-level mismatch: got %+v want %+v", got, manifest)
	}
	if got.Artifacts[AuditArtifactLlamaServer].SHA256 != "aaa" {
		t.Errorf("llama_server sha = %q", got.Artifacts[AuditArtifactLlamaServer].SHA256)
	}
	if got.Artifacts[AuditArtifactModel].URL != "https://example/qwen3.gguf" {
		t.Errorf("model url = %q", got.Artifacts[AuditArtifactModel].URL)
	}
}

func TestReadInstalledManifest_NotInstalled(t *testing.T) {
	paths := newAuditTestPaths(t)

	_, err := ReadInstalledManifest(paths)
	if !os.IsNotExist(err) {
		t.Errorf("err = %v, want os.IsNotExist (no install has run)", err)
	}
}

func TestRecordInstall_OverwritesAtomically(t *testing.T) {
	paths := newAuditTestPaths(t)

	// Initial installation
	if err := recordInstall(paths, "https://m", 1, "", map[string]InstalledArtifact{
		AuditArtifactLlamaServer: {URL: "first", SHA256: "1"},
	}); err != nil {
		t.Fatalf("recordInstall #1: %v", err)
	}

	// Update (or re-install)
	if err := recordInstall(paths, "https://m", 1, "", map[string]InstalledArtifact{
		AuditArtifactLlamaServer: {URL: "second", SHA256: "2"},
	}); err != nil {
		t.Fatalf("recordInstall #2: %v", err)
	}

	got, err := ReadInstalledManifest(paths)
	if err != nil {
		t.Fatalf("ReadInstalledManifest: %v", err)
	}
	if got.Artifacts[AuditArtifactLlamaServer].URL != "second" {
		t.Errorf("URL = %q, want second", got.Artifacts[AuditArtifactLlamaServer].URL)
	}
}

// recordInstall must preserve a previously-written artifact when a partial
// install only knows about one slot. EnsureLlamaServer followed by
// EnsureModel must not wipe the llama_server entry.
func TestRecordInstall_PreservesExistingArtifacts(t *testing.T) {
	paths := newAuditTestPaths(t)

	// First install: llama_server only
	if err := recordInstall(paths, "https://m1", 1, "", map[string]InstalledArtifact{
		AuditArtifactLlamaServer: {URL: "u-llama", SHA256: "s-llama", Path: "/p/llama"},
	}); err != nil {
		t.Fatalf("recordInstall #1: %v", err)
	}

	// Second install: model only
	if err := recordInstall(paths, "https://m2", 1, "qwen", map[string]InstalledArtifact{
		AuditArtifactModel: {URL: "u-model", SHA256: "s-model", Path: "/p/model"},
	}); err != nil {
		t.Fatalf("recordInstall #2: %v", err)
	}

	got, err := ReadInstalledManifest(paths)
	if err != nil {
		t.Fatalf("ReadInstalledManifest: %v", err)
	}

	if got.ManifestURL != "https://m2" {
		t.Errorf("manifest_url = %q, want last write", got.ManifestURL)
	}
	if got.ModelVariant != "qwen" {
		t.Errorf("model_variant = %q", got.ModelVariant)
	}

	// recordInstall must preserve previously written artifact
	if a, ok := got.Artifacts[AuditArtifactLlamaServer]; !ok || a.URL != "u-llama" {
		t.Errorf("llama_server entry lost or wrong: %+v", a)
	}
	if a, ok := got.Artifacts[AuditArtifactModel]; !ok || a.URL != "u-model" {
		t.Errorf("model entry missing: %+v", a)
	}
}

func TestRecordInstall_RecoversFromCorruptFile(t *testing.T) {
	paths := newAuditTestPaths(t)

	if err := os.WriteFile(InstalledManifestPath(paths), []byte("{not json"), 0o600); err != nil {
		t.Fatalf("seed corrupt file: %v", err)
	}

	if err := recordInstall(paths, "https://m", 1, "v", map[string]InstalledArtifact{
		AuditArtifactModel: {URL: "u", SHA256: "s", Path: "/p"},
	}); err != nil {
		t.Fatalf("recordInstall: %v", err)
	}

	got, err := ReadInstalledManifest(paths)
	if err != nil {
		t.Fatalf("ReadInstalledManifest: %v", err)
	}
	if got.Artifacts[AuditArtifactModel].URL != "u" {
		t.Errorf("post-recovery URL = %q", got.Artifacts[AuditArtifactModel].URL)
	}
}

func TestInstalledManifest_JSONShape(t *testing.T) {
	paths := newAuditTestPaths(t)

	manifest := &InstalledManifest{
		ManifestURL:     "https://m",
		ManifestVersion: 1,
		Platform:        "linux-amd64",
		InstalledAt:     "2026-01-01T00:00:00Z",
		ModelVariant:    "v",
		Artifacts: map[string]InstalledArtifact{
			AuditArtifactLlamaServer: {URL: "u", SHA256: "s", Path: "/p", Size: 9},
		},
	}
	if err := WriteInstalledManifest(paths, manifest); err != nil {
		t.Fatalf("Write: %v", err)
	}

	data, err := os.ReadFile(InstalledManifestPath(paths))
	if err != nil {
		t.Fatal(err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}

	for _, k := range []string{"manifest_url", "manifest_version", "platform", "installed_at", "model_variant", "artifacts"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("missing top-level key %q", k)
		}
	}

	arts, ok := raw["artifacts"].(map[string]any)
	if !ok {
		t.Fatal("artifacts not an object")
	}
	llama, ok := arts["llama_server"].(map[string]any)
	if !ok {
		t.Fatal("artifacts.llama_server missing")
	}

	for _, k := range []string{"url", "sha256", "path", "size"} {
		if _, ok := llama[k]; !ok {
			t.Errorf("artifacts.llama_server.%s missing", k)
		}
	}
}
