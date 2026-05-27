package spawn

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeFile is a small helper that creates and (optionally) marks a file
// executable. Returns the path.
func writeFile(t *testing.T, dir, name string, contents []byte, executable bool) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, contents, 0o644); err != nil {
		t.Fatalf("write %s: %v", p, err)
	}
	if executable {
		if err := os.Chmod(p, 0o755); err != nil {
			t.Fatalf("chmod %s: %v", p, err)
		}
	}
	return p
}

// writeValidGGUF writes a stub GGUF file (just the magic + filler) at the
// given path and returns it.
func writeValidGGUF(t *testing.T, dir, name string) string {
	t.Helper()
	body := append([]byte("GGUF"), make([]byte, 16)...)
	return writeFile(t, dir, name, body, false)
}

func TestLoadConfig_ValidConfig(t *testing.T) {
	dir := t.TempDir()
	llamaBin := writeFile(t, dir, "llama-server", []byte("#!/bin/sh\n"), true)
	runedBin := writeFile(t, dir, "runed", []byte("#!/bin/sh\n"), true)
	model := writeValidGGUF(t, dir, "model.gguf")
	cfgPath := writeFile(t, dir, "config.json", []byte(`{
        "version": 1,
        "llama_server": "`+llamaBin+`",
        "model": "`+model+`",
        "runed_binary": "`+runedBin+`",
        "idle_timeout": "5m"
    }`), false)
	t.Setenv("RUNED_CONFIG", cfgPath)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LlamaServer != llamaBin {
		t.Errorf("LlamaServer = %q; want %q", cfg.LlamaServer, llamaBin)
	}
	if cfg.Model != model {
		t.Errorf("Model = %q; want %q", cfg.Model, model)
	}
	if cfg.RunedBinary != runedBin {
		t.Errorf("RunedBinary = %q; want %q", cfg.RunedBinary, runedBin)
	}
	if cfg.IdleTimeout != "5m" {
		t.Errorf("IdleTimeout = %q; want %q", cfg.IdleTimeout, "5m")
	}
}

func TestLoadConfig_VersionMismatch(t *testing.T) {
	dir := t.TempDir()
	llamaBin := writeFile(t, dir, "llama-server", []byte("x"), true)
	runedBin := writeFile(t, dir, "runed", []byte("x"), true)
	model := writeValidGGUF(t, dir, "model.gguf")
	cfgPath := writeFile(t, dir, "config.json", []byte(`{
        "version": 2,
        "llama_server": "`+llamaBin+`",
        "model": "`+model+`",
        "runed_binary": "`+runedBin+`"
    }`), false)
	t.Setenv("RUNED_CONFIG", cfgPath)
	_, err := LoadConfig()
	if err == nil || !strings.Contains(err.Error(), "version 2") {
		t.Fatalf("expected version mismatch, got: %v", err)
	}
}

func TestLoadConfig_MissingLlamaServer(t *testing.T) {
	dir := t.TempDir()
	runedBin := writeFile(t, dir, "runed", []byte("x"), true)
	model := writeValidGGUF(t, dir, "model.gguf")
	cfgPath := writeFile(t, dir, "config.json", []byte(`{
        "version": 1,
        "llama_server": "`+filepath.Join(dir, "does-not-exist")+`",
        "model": "`+model+`",
        "runed_binary": "`+runedBin+`"
    }`), false)
	t.Setenv("RUNED_CONFIG", cfgPath)
	_, err := LoadConfig()
	if err == nil || !strings.Contains(err.Error(), "llama_server") {
		t.Fatalf("expected missing llama_server error, got: %v", err)
	}
}

func TestLoadConfig_ModelNotGGUF(t *testing.T) {
	dir := t.TempDir()
	llamaBin := writeFile(t, dir, "llama-server", []byte("x"), true)
	runedBin := writeFile(t, dir, "runed", []byte("x"), true)
	model := writeFile(t, dir, "model.gguf", []byte("NOTGGUF"), false)
	cfgPath := writeFile(t, dir, "config.json", []byte(`{
        "version": 1,
        "llama_server": "`+llamaBin+`",
        "model": "`+model+`",
        "runed_binary": "`+runedBin+`"
    }`), false)
	t.Setenv("RUNED_CONFIG", cfgPath)
	_, err := LoadConfig()
	if err == nil || !strings.Contains(err.Error(), "not a GGUF") {
		t.Fatalf("expected GGUF magic error, got: %v", err)
	}
}

// TestLoadConfig_MissingFile_NoFallback: with no config file, no PATH
// match, and no DefaultRunedBinary baked in, LoadConfig should fail with
// the unresolved-runed_binary error.
func TestLoadConfig_MissingFile_NoFallback(t *testing.T) {
	t.Setenv("RUNED_CONFIG", "/tmp/definitely-does-not-exist-runed-config.json")
	t.Setenv("PATH", "/nonexistent")
	saved := DefaultRunedBinary
	DefaultRunedBinary = ""
	t.Cleanup(func() { DefaultRunedBinary = saved })

	_, err := LoadConfig()
	if err == nil || !strings.Contains(err.Error(), "unresolved") {
		t.Fatalf("expected runed_binary unresolved error, got: %v", err)
	}
}

// TestLoadConfig_MissingFile_UsesDefaultRunedBinary: with no config file
// but DefaultRunedBinary baked in (the release/dev build case), LoadConfig
// returns a valid config with LlamaServer/Model unset — the daemon will
// self-bootstrap those.
func TestLoadConfig_MissingFile_UsesDefaultRunedBinary(t *testing.T) {
	dir := t.TempDir()
	runedBin := writeFile(t, dir, "runed", []byte("#!/bin/sh\n"), true)

	t.Setenv("RUNED_CONFIG", filepath.Join(dir, "nope.json"))
	t.Setenv("PATH", "/nonexistent")
	saved := DefaultRunedBinary
	DefaultRunedBinary = runedBin
	t.Cleanup(func() { DefaultRunedBinary = saved })

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LlamaServer != "" || cfg.Model != "" {
		t.Errorf("expected empty LlamaServer/Model (self-bootstrap), got %q/%q",
			cfg.LlamaServer, cfg.Model)
	}
	if cfg.RunedBinary != runedBin {
		t.Errorf("RunedBinary = %q; want %q", cfg.RunedBinary, runedBin)
	}
}

// TestLoadConfig_OptionalLlamaServerAndModel: a config that sets only
// runed_binary and omits llama_server/model parses successfully — those
// fields delegate to the daemon's self-bootstrap.
func TestLoadConfig_OptionalLlamaServerAndModel(t *testing.T) {
	dir := t.TempDir()
	runedBin := writeFile(t, dir, "runed", []byte("#!/bin/sh\n"), true)
	cfgPath := writeFile(t, dir, "config.json", []byte(`{
        "version": 1,
        "runed_binary": "`+runedBin+`"
    }`), false)
	t.Setenv("RUNED_CONFIG", cfgPath)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LlamaServer != "" || cfg.Model != "" {
		t.Errorf("expected empty LlamaServer/Model, got %q/%q",
			cfg.LlamaServer, cfg.Model)
	}
	if cfg.RunedBinary != runedBin {
		t.Errorf("RunedBinary = %q; want %q", cfg.RunedBinary, runedBin)
	}
}
