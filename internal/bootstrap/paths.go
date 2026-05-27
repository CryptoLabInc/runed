// Package bootstrap downloads the llama-server binary and embedding model
// on the daemon's first boot so runed can run standalone — no pre-installed
// dependencies, no external setup scripts.
//
// The manifest URL comes from RUNED_MANIFEST. runed deliberately does not
// reuse rune install's manifest: their schemas are independent (rune ships
// a `runed_bundles` map; runed expects `platforms[*].llama_server`,
// `models`, `default_model`), so a shared URL can't satisfy both readers.
package bootstrap

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

const (
	EnvHome         = "RUNED_HOME"
	EnvManifest     = "RUNED_MANIFEST"
	EnvModelVariant = "RUNED_MODEL_VARIANT"
)

// Paths is the on-disk layout under $RUNED_HOME (default ~/.runed).
//
//	~/.runed/
//	├── bin/
//	│   ├── runed                          (placed by rune install)
//	│   └── llama-cpp/                     (tarball extracted here)
//	│       └── <manifest.exec>            e.g. build/bin/llama-server
//	├── models/
//	│   └── <variant>.gguf
//	├── cache/                             (temp downloads, cleaned post-extract)
//	├── config.json                        (runtime config: model_variant, ...)
//	├── install.lock                       (flock — serializes self-bootstrap)
//	└── logs/
type Paths struct {
	Home        string
	Bin         string
	LlamaDir    string
	Models      string
	Config      string
	InstallLock string
	Logs        string
	Cache       string
}

func Resolve() (*Paths, error) {
	home := os.Getenv(EnvHome)
	if home == "" {
		u, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("user home: %w", err)
		}
		home = filepath.Join(u, ".runed")
	}
	abs, err := filepath.Abs(home)
	if err != nil {
		return nil, fmt.Errorf("abs %s: %w", home, err)
	}
	bin := filepath.Join(abs, "bin")
	return &Paths{
		Home:        abs,
		Bin:         bin,
		LlamaDir:    filepath.Join(bin, "llama-cpp"),
		Models:      filepath.Join(abs, "models"),
		Config:      filepath.Join(abs, "config.json"),
		InstallLock: filepath.Join(abs, "install.lock"),
		Logs:        filepath.Join(abs, "logs"),
		Cache:       filepath.Join(abs, "cache"),
	}, nil
}

func (p *Paths) EnsureDirs() error {
	for _, d := range []string{p.Bin, p.LlamaDir, p.Models, p.Logs, p.Cache} {
		if err := os.MkdirAll(d, 0o700); err != nil {
			return fmt.Errorf("mkdir %s: %w", d, err)
		}
	}
	return nil
}

// ModelPath returns the on-disk filename for a model variant ID. The ID
// is the manifest key (e.g. "qwen3-embedding-0.6b.q6_K"); on-disk we
// append ".gguf" so a glob in models/ surfaces only model files.
func (p *Paths) ModelPath(variant string) string {
	return filepath.Join(p.Models, variant+".gguf")
}

func PlatformTuple() string {
	return runtime.GOOS + "-" + runtime.GOARCH
}
