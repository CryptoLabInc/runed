package bootstrap

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig_Missing(t *testing.T) {
	c, err := LoadConfig(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("LoadConfig missing: %v", err)
	}
	if c.Version != ConfigVersion {
		t.Errorf("default Version: got %d, want %d", c.Version, ConfigVersion)
	}
	if c.ModelVariant != "" {
		t.Errorf("default ModelVariant: got %q, want empty", c.ModelVariant)
	}
}

func TestLoadConfig_Valid(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	data := []byte(`{"version": 1, "model_variant": "qwen3-embedding-0.6b.q8_0"}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	c, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.Version != 1 {
		t.Errorf("Version: got %d", c.Version)
	}
	if c.ModelVariant != "qwen3-embedding-0.6b.q8_0" {
		t.Errorf("ModelVariant: got %q", c.ModelVariant)
	}
}

func TestLoadConfig_VersionMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{"version": 99}`), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadConfig(path); err == nil {
		t.Fatal("expected version mismatch error")
	}
}

// TestLoadConfig_IgnoresSpawnFields confirms the file can carry the
// spawn schema (LlamaServer, Model, RunedBinary, ...) without bootstrap
// blowing up — both readers share the same config.json.
func TestLoadConfig_IgnoresSpawnFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	data := []byte(`{
		"version": 1,
		"llama_server": "/tmp/x",
		"model": "/tmp/y.gguf",
		"runed_binary": "/tmp/runed",
		"model_variant": "qwen3-embedding-0.6b.q6_K"
	}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	c, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.ModelVariant != "qwen3-embedding-0.6b.q6_K" {
		t.Errorf("ModelVariant: got %q", c.ModelVariant)
	}
}
