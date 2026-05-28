package bootstrap

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

// InstallLockTimeout bounds how long a concurrent boot will wait for the
// install.lock holder. The leader can run ensureLlamaServer (bounded by
// downloadTimeout) followed by ensureModel (bounded by downloadTimeout
// again), so the worst-case legitimate hold is 2*downloadTimeout. A
// shorter timeout would fail trailers during slow-but-healthy installs
// even though the leader is making progress.
//
// The "wedged forever" guarantee still holds: downloadTimeout caps any
// single artifact transfer on the leader side, so a longer lock timeout
// doesn't reintroduce zombie risk.
const InstallLockTimeout = 2 * downloadTimeout

// progressLogInterval is the minimum gap between progress log lines for
// a single download. Tuned high enough that a tiny manifest fetch only
// produces one line, but low enough that a slow model download still
// shows movement.
const progressLogInterval = 2 * time.Second

// EnsureAll resolves the model variant and ensures both llama-server and
// the model GGUF are present, downloading any missing pieces while
// holding $RUNED_HOME/install.lock. Returns the absolute path to the
// llama-server executable and the GGUF file the daemon should load.
//
// logger may be nil; slog.Default() is used in that case. All progress
// and per-step status is emitted through this logger — callers don't
// need to thread a separate ProgressFunc.
func EnsureAll(ctx context.Context, p *Paths, m *Manifest, logger *slog.Logger) (llamaBin, modelPath, variant string, err error) {
	if logger == nil {
		logger = slog.Default()
	}
	if err := p.EnsureDirs(); err != nil {
		return "", "", "", err
	}
	variant, err = ResolveModelVariant(p, m)
	if err != nil {
		return "", "", "", err
	}
	logger.Info("ensure: resolved model variant",
		"variant", variant,
		"default_model", m.DefaultModel)

	lock, err := AcquireLock(p.InstallLock, InstallLockTimeout)
	if err != nil {
		return "", "", "", fmt.Errorf("install lock: %w", err)
	}
	defer lock.Release()

	llamaBin, err = ensureLlamaServer(ctx, p, m, logger)
	if err != nil {
		return "", "", "", err
	}
	modelPath, err = ensureModel(ctx, p, m, variant, logger)
	if err != nil {
		return "", "", "", err
	}
	return llamaBin, modelPath, variant, nil
}

// makeProgress returns a throttled ProgressFunc that logs at most one
// line per progressLogInterval, plus a final 100% line on completion.
// total ≤ 0 means Content-Length wasn't advertised; the function falls
// back to byte-count-only output.
func makeProgress(logger *slog.Logger, stage string, expectedTotal int64) ProgressFunc {
	var lastReport time.Time
	return func(downloaded, observedTotal int64) {
		total := expectedTotal
		if total <= 0 {
			total = observedTotal
		}
		complete := total > 0 && downloaded >= total
		if !complete && time.Since(lastReport) < progressLogInterval {
			return
		}
		lastReport = time.Now()
		attrs := []any{"stage", stage, "downloaded", downloaded}
		if total > 0 {
			pct := float64(downloaded) / float64(total) * 100
			attrs = append(attrs, "total", total, "pct", fmt.Sprintf("%.1f%%", pct))
		}
		logger.Info("download progress", attrs...)
	}
}

// ResolveModelVariant picks the model variant ID using priority:
//
//	RUNED_MODEL_VARIANT env > config.json model_variant > manifest.default_model
func ResolveModelVariant(p *Paths, m *Manifest) (string, error) {
	if v := os.Getenv(EnvModelVariant); v != "" {
		return v, nil
	}
	cfg, err := LoadConfig(p.Config)
	if err != nil {
		return "", err
	}
	if cfg.ModelVariant != "" {
		return cfg.ModelVariant, nil
	}
	if m.DefaultModel != "" {
		return m.DefaultModel, nil
	}
	return "", errors.New("model variant not specified: set RUNED_MODEL_VARIANT, config.model_variant, or manifest.default_model")
}

// ensureLlamaServer returns the path to the llama-server executable,
// extracting or downloading the artifact as needed. A sidecar marker
// file (.llama_server.sha256) tracks the last-installed tarball hash so
// repeat boots don't re-extract.
func ensureLlamaServer(ctx context.Context, p *Paths, m *Manifest, logger *slog.Logger) (string, error) {
	spec, err := m.LlamaServerForCurrentPlatform()
	if err != nil {
		return "", err
	}
	target := llamaServerTarget(p, spec)
	logger.Info("ensure llama_server: target",
		"platform", PlatformTuple(),
		"exec", spec.Exec,
		"extract", spec.Extract,
		"target", target,
		"want_sha256", spec.SHA256,
		"size", spec.Size)

	marker := filepath.Join(p.LlamaDir, ".llama_server.sha256")
	if existing, rerr := os.ReadFile(marker); rerr == nil && string(existing) == spec.SHA256 {
		if _, serr := os.Stat(target); serr == nil {
			logger.Info("ensure llama_server: cache hit, skipping download",
				"marker", marker)
			return target, nil
		}
		logger.Info("ensure llama_server: marker matches but target missing, redoing install",
			"target", target)
	}

	progress := makeProgress(logger, "llama_server", spec.Size)
	switch spec.Extract {
	case "":
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return "", fmt.Errorf("ensure llama_server: mkdir: %w", err)
		}
		logger.Info("ensure llama_server: downloading raw binary", "url", spec.URL)
		if err := DownloadAndVerify(ctx, spec.URL, spec.SHA256, spec.Size, target, progress); err != nil {
			return "", fmt.Errorf("ensure llama_server: download: %w", err)
		}
		if err := os.Chmod(target, 0o755); err != nil {
			return "", fmt.Errorf("ensure llama_server: chmod: %w", err)
		}
	case "tar.gz":
		tarPath := filepath.Join(p.Cache, "llama-server.tar.gz")
		logger.Info("ensure llama_server: downloading tarball",
			"url", spec.URL,
			"cache", tarPath)
		if err := DownloadAndVerify(ctx, spec.URL, spec.SHA256, spec.Size, tarPath, progress); err != nil {
			return "", fmt.Errorf("ensure llama_server: download: %w", err)
		}
		defer os.Remove(tarPath)
		logger.Info("ensure llama_server: extracting tarball", "dest", p.LlamaDir)
		extracted, err := ExtractTarGz(tarPath, p.LlamaDir)
		if err != nil {
			return "", fmt.Errorf("ensure llama_server: extract: %w", err)
		}
		logger.Info("ensure llama_server: extracted", "files", len(extracted))
	default:
		return "", fmt.Errorf("manifest: unsupported extract type %q", spec.Extract)
	}

	if _, err := os.Stat(target); err != nil {
		return "", fmt.Errorf("ensure llama_server: exec missing after install: %s: %w", target, err)
	}
	if err := os.Chmod(target, 0o755); err != nil {
		return "", fmt.Errorf("ensure llama_server: chmod target: %w", err)
	}
	// Marker write is an optimization; tolerate failure (we'll just re-extract next boot).
	_ = os.WriteFile(marker, []byte(spec.SHA256), 0o600)
	logger.Info("ensure llama_server: install complete", "target", target)
	return target, nil
}

// llamaServerTarget computes the on-disk path of the executable after
// install. For raw binaries (extract=""), Exec is optional and defaults
// to "llama-server" placed directly under bin/llama-cpp/.
func llamaServerTarget(p *Paths, spec *LlamaServerSpec) string {
	exec := spec.Exec
	if exec == "" {
		exec = "llama-server"
	}
	return filepath.Join(p.LlamaDir, filepath.FromSlash(exec))
}

func ensureModel(ctx context.Context, p *Paths, m *Manifest, variant string, logger *slog.Logger) (string, error) {
	spec, err := m.ModelSpec(variant)
	if err != nil {
		return "", err
	}
	target := p.ModelPath(variant)
	logger.Info("ensure model: target",
		"variant", variant,
		"target", target,
		"want_sha256", spec.SHA256,
		"size", spec.Size)

	ok, err := FileMatchesSHA256(target, spec.SHA256)
	if err != nil {
		return "", fmt.Errorf("ensure model: hash existing: %w", err)
	}
	if ok {
		logger.Info("ensure model: cache hit, skipping download")
		return target, nil
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return "", err
	}
	logger.Info("ensure model: downloading GGUF", "url", spec.URL)
	progress := makeProgress(logger, "model", spec.Size)
	if err := DownloadAndVerify(ctx, spec.URL, spec.SHA256, spec.Size, target, progress); err != nil {
		return "", fmt.Errorf("ensure model: download: %w", err)
	}
	logger.Info("ensure model: install complete", "target", target)
	return target, nil
}
