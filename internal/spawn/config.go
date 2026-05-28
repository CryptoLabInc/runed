// Package spawn handles auto-spawning of the runed daemon from the client.
// It reads ~/.runed/config.json for optional overrides — none of its fields
// are required. When LlamaServer/Model aren't set, the daemon does its own
// self-bootstrap (see internal/bootstrap). When RunedBinary isn't set, the
// package looks it up in PATH and then falls back to a build-time default.
//
// The package is unix-only for Plan B. Windows support is deferred.
package spawn

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

// ConfigVersion is the only supported value of Config.Version.
const ConfigVersion = 1

// DefaultRunedBinary is the fallback path for the runed daemon binary
// when neither config nor PATH resolves one. Empty in source; release/dev
// builds inject the production path via ldflags so end users don't need
// to set anything:
//
//	go build -ldflags "-X github.com/CryptoLabInc/runed/internal/spawn.DefaultRunedBinary=$HOME/.runed/bin/runed"
//
// or via the Makefile:
//
//	make build DEFAULT_RUNED_BINARY=$HOME/.runed/bin/runed
var DefaultRunedBinary = ""

// Config is the parsed shape of ~/.runed/config.json. All fields are
// optional; the spawn flow resolves missing values from sensible defaults
// (PATH lookup, build-time injection, daemon self-bootstrap).
//
// Note: the same config.json is read by internal/bootstrap with a different
// schema (model_variant). We do NOT DisallowUnknownFields so the two
// readers can coexist on one file without arguing over fields they don't
// each own.
type Config struct {
	Version     int    `json:"version"`
	LlamaServer string `json:"llama_server,omitempty"`
	Model       string `json:"model,omitempty"`
	RunedBinary string `json:"runed_binary,omitempty"`
	IdleTimeout string `json:"idle_timeout,omitempty"`
}

// LoadConfig reads, parses, and validates the config file. A missing file
// is treated as "no overrides" — equivalent to a default Config with
// every field unset. Resolved fields are canonicalized to absolute paths
// before return, so the detached daemon child sees the same files
// regardless of the caller's working directory.
func LoadConfig() (*Config, error) {
	path, err := resolveConfigPath()
	if err != nil {
		return nil, err
	}
	cfg := &Config{Version: ConfigVersion}
	data, readErr := os.ReadFile(path)
	if readErr != nil && !errors.Is(readErr, os.ErrNotExist) {
		return nil, fmt.Errorf("read config %s: %w", path, readErr)
	}
	if readErr == nil {
		if err := json.NewDecoder(bytes.NewReader(data)).Decode(cfg); err != nil {
			return nil, fmt.Errorf("config parse %s: %w", path, err)
		}
		if cfg.Version != ConfigVersion {
			return nil, fmt.Errorf("config %s: version %d not supported (need %d)",
				path, cfg.Version, ConfigVersion)
		}
	}
	if err := resolveRunedBinary(cfg); err != nil {
		return nil, err
	}
	// LlamaServer / Model are only validated when set. An empty value means
	// "let the daemon's self-bootstrap pick the artifact" — launchDaemon
	// won't propagate the empty env var.
	if cfg.LlamaServer != "" {
		if err := validateExecutable(cfg.LlamaServer, "llama_server"); err != nil {
			return nil, err
		}
		if cfg.LlamaServer, err = filepath.Abs(cfg.LlamaServer); err != nil {
			return nil, fmt.Errorf("abs llama_server: %w", err)
		}
	}
	if cfg.Model != "" {
		if err := validateGGUF(cfg.Model); err != nil {
			return nil, err
		}
		if cfg.Model, err = filepath.Abs(cfg.Model); err != nil {
			return nil, fmt.Errorf("abs model: %w", err)
		}
	}
	if err := validateExecutable(cfg.RunedBinary, "runed_binary"); err != nil {
		return nil, err
	}
	if cfg.RunedBinary, err = filepath.Abs(cfg.RunedBinary); err != nil {
		return nil, fmt.Errorf("abs runed_binary: %w", err)
	}
	return cfg, nil
}

// resolveRunedBinary fills cfg.RunedBinary using the first source that
// yields a usable path. Order matters:
//
//  1. config field (explicit caller override)
//  2. PATH lookup (`runed` on PATH — most portable)
//  3. DefaultRunedBinary (build-time injection — release / dev default)
//  4. os.Executable() iff basename == "runed" (self-spawn from runed itself,
//     not from a client like rundemo whose Executable would be `rundemo`)
//
// We deliberately demote os.Executable() below the build-time default
// because client binaries (rundemo, rune-cli) routinely call this code,
// and grabbing their own path would point spawn at the wrong binary.
func resolveRunedBinary(cfg *Config) error {
	if cfg.RunedBinary != "" {
		return nil
	}
	if p, err := exec.LookPath("runed"); err == nil {
		cfg.RunedBinary = p
		return nil
	}
	if DefaultRunedBinary != "" {
		cfg.RunedBinary = DefaultRunedBinary
		return nil
	}
	if self, err := os.Executable(); err == nil && filepath.Base(self) == "runed" {
		cfg.RunedBinary = self
		return nil
	}
	return errors.New(
		"runed_binary unresolved: set config.runed_binary, place `runed` on PATH, " +
			"or rebuild with -ldflags '-X .../spawn.DefaultRunedBinary=...'")
}

func resolveConfigPath() (string, error) {
	if p := os.Getenv("RUNED_CONFIG"); p != "" {
		return p, nil
	}
	if h := os.Getenv("RUNED_HOME"); h != "" {
		return filepath.Join(h, "config.json"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("user home: %w", err)
	}
	return filepath.Join(home, ".runed", "config.json"), nil
}

func validateExecutable(path, label string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s %q: %w", label, path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("%s %q: is a directory", label, path)
	}
	if info.Mode()&0o111 == 0 {
		return fmt.Errorf("%s %q: not executable", label, path)
	}
	return nil
}

func validateGGUF(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("model %q: %w", path, err)
	}
	defer f.Close()
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return fmt.Errorf("model %q: read magic: %w", path, err)
	}
	if string(magic[:]) != "GGUF" {
		return fmt.Errorf("model %q: not a GGUF file (magic=%x)", path, magic)
	}
	return nil
}
