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

// maxDownloadAttempts is the cap on retries for a single artifact.
// A ~470MB model over a flaky connection benefits from a couple of
// retries; beyond that, the failure is most likely a manifest/server
// mismatch (wrong SHA, missing file) where retrying just wastes time.
const maxDownloadAttempts = 3

// downloadRetryBackoff is the initial wait between attempts; it's
// multiplied by retryBackoffMultiplier each subsequent retry so a
// transient blip recovers quickly while a server-side cold-start has
// time to warm up. Declared as var so tests can compress real waits.
var downloadRetryBackoff = 5 * time.Second

const retryBackoffMultiplier = 3

// StatusReporter receives throttled progress ticks from Ensure* tagged
// with stage ("llama_server" or "model"). bytesTotal == 0 means the
// total isn't yet known (no Content-Length observed). nil is a valid
// "no sink" value; reporter calls run inline on the download goroutine
// and should return quickly.
type StatusReporter func(stage string, bytesDone, bytesTotal int64)

// EnsureAll ensures both llama-server and the model GGUF are installed
// under a single lock acquisition. Use when both RUNED_LLAMA_SERVER and
// RUNED_MODEL are unset; for the partial-set case, call
// EnsureLlamaServer / EnsureModel individually so the env-overridden
// side isn't redownloaded only to be discarded.
//
// logger may be nil (slog.Default used). reporter may be nil.
func EnsureAll(ctx context.Context, p *Paths, m *Manifest, logger *slog.Logger, reporter StatusReporter) (llamaBinPath, modelPath, variant string, err error) {
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

	// Stage tick before AcquireLock so a lock-waiting trailer surfaces
	// the right Phase while it waits.
	if reporter != nil {
		reporter("llama_server", 0, 0)
	}
	lock, err := AcquireLock(p.InstallLock, InstallLockTimeout)
	if err != nil {
		return "", "", "", fmt.Errorf("install lock: %w", err)
	}
	defer lock.Release()

	llamaBinPath, llamaSpec, err := ensureLlamaServer(ctx, p, m, logger, reporter)
	if err != nil {
		return "", "", "", err
	}
	if reporter != nil {
		reporter("model", 0, 0)
	}
	modelPath, modelSpec, err := ensureModel(ctx, p, m, variant, logger, reporter)
	if err != nil {
		return "", "", "", err
	}

	// Audit installation
	auditArtifacts := map[string]InstalledArtifact{
		AuditArtifactLlamaServer: {URL: llamaSpec.URL, SHA256: llamaSpec.SHA256, Path: llamaBinPath, Size: statSize(llamaBinPath, llamaSpec.Size)},
		AuditArtifactModel:       {URL: modelSpec.URL, SHA256: modelSpec.SHA256, Path: modelPath, Size: statSize(modelPath, modelSpec.Size)},
	}
	if auditErr := recordInstall(p, ResolveManifestURL(), m.Version, variant, auditArtifacts); auditErr != nil {
		logger.Warn("audit: installed.json write failed", "err", auditErr)
	}

	return llamaBinPath, modelPath, variant, nil
}

// EnsureLlamaServer ensures the llama-server binary is installed. The
// manifest must have an entry for the current platform (this is the
// caller's intent — they explicitly asked for the manifest's binary).
func EnsureLlamaServer(ctx context.Context, p *Paths, m *Manifest, logger *slog.Logger, reporter StatusReporter) (string, error) {
	if logger == nil {
		logger = slog.Default()
	}
	if err := p.EnsureDirs(); err != nil {
		return "", err
	}
	if reporter != nil {
		reporter("llama_server", 0, 0)
	}
	lock, err := AcquireLock(p.InstallLock, InstallLockTimeout)
	if err != nil {
		return "", fmt.Errorf("install lock: %w", err)
	}
	defer lock.Release()

	llamaBinPath, llamaSpec, err := ensureLlamaServer(ctx, p, m, logger, reporter)
	if err != nil {
		return "", err
	}

	if auditErr := recordInstall(p, ResolveManifestURL(), m.Version, "", map[string]InstalledArtifact{
		AuditArtifactLlamaServer: {URL: llamaSpec.URL, SHA256: llamaSpec.SHA256, Path: llamaBinPath, Size: statSize(llamaBinPath, llamaSpec.Size)},
	}); auditErr != nil {
		logger.Warn("audit: installed.json write failed", "err", auditErr)
	}

	return llamaBinPath, nil
}

// EnsureModel ensures the model GGUF is installed. Skips
// LlamaServerForCurrentPlatform so a caller on a platform missing from
// the manifest can still bootstrap a model when RUNED_LLAMA_SERVER
// points at a working binary.
func EnsureModel(ctx context.Context, p *Paths, m *Manifest, logger *slog.Logger, reporter StatusReporter) (modelPath, variant string, err error) {
	if logger == nil {
		logger = slog.Default()
	}
	if err := p.EnsureDirs(); err != nil {
		return "", "", err
	}
	variant, err = ResolveModelVariant(p, m)
	if err != nil {
		return "", "", err
	}
	logger.Info("ensure: resolved model variant",
		"variant", variant,
		"default_model", m.DefaultModel)

	if reporter != nil {
		reporter("model", 0, 0)
	}
	lock, err := AcquireLock(p.InstallLock, InstallLockTimeout)
	if err != nil {
		return "", "", fmt.Errorf("install lock: %w", err)
	}
	defer lock.Release()

	modelPath, modelSpec, err := ensureModel(ctx, p, m, variant, logger, reporter)
	if err != nil {
		return "", "", err
	}

	if auditErr := recordInstall(p, ResolveManifestURL(), m.Version, variant, map[string]InstalledArtifact{
		AuditArtifactModel: {URL: modelSpec.URL, SHA256: modelSpec.SHA256, Path: modelPath, Size: statSize(modelPath, modelSpec.Size)},
	}); auditErr != nil {
		logger.Warn("audit: installed.json write failed (non-fatal)", "err", auditErr)
	}

	return modelPath, variant, nil
}

// downloadWithRetry wraps DownloadAndVerify with bounded exponential
// backoff. Multi-hundred-MB downloads are vulnerable to transient
// network blips that an immediate single-attempt boot would surface as
// a hard failure to whichever supervisor (rune spawn, systemd) launched
// runed. Caller-driven cancellation (ctx.Err) skips the retry path so
// shutdown isn't delayed.
func downloadWithRetry(ctx context.Context, url, sha string, size int64, dest string, progress ProgressFunc, logger *slog.Logger, stage string) error {
	var lastErr error
	backoff := downloadRetryBackoff
	for attempt := 1; attempt <= maxDownloadAttempts; attempt++ {
		if attempt > 1 {
			logger.Warn("retrying download",
				"stage", stage,
				"attempt", attempt,
				"max", maxDownloadAttempts,
				"after", lastErr,
				"backoff", backoff.String())
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return fmt.Errorf("download retry aborted: %w", ctx.Err())
			}
			backoff *= retryBackoffMultiplier
		}
		err := DownloadAndVerify(ctx, url, sha, size, dest, progress)
		if err == nil {
			return nil
		}
		// Caller cancelled — don't burn retries on a shutdown.
		if ctx.Err() != nil {
			return err
		}
		lastErr = err
	}
	return fmt.Errorf("download failed after %d attempts: %w", maxDownloadAttempts, lastErr)
}

// makeProgress returns a throttled ProgressFunc that logs at most one
// line per progressLogInterval and, if reporter != nil, forwards the
// same throttled tick. total ≤ 0 means Content-Length wasn't advertised;
// falls back to byte-count-only output.
func makeProgress(logger *slog.Logger, reporter StatusReporter, stage string, expectedTotal int64) ProgressFunc {
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
		if reporter != nil {
			reporter(stage, downloaded, total)
		}
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

// ensureLlamaServer returns the path to the llama-server executable and manifest spec,
// extracting or downloading the artifact as needed. A sidecar marker
// file (.llama_server.sha256) tracks the last-installed tarball hash so
// repeat boots don't re-extract.
func ensureLlamaServer(ctx context.Context, p *Paths, m *Manifest, logger *slog.Logger, reporter StatusReporter) (string, *LlamaServerSpec, error) {
	// Caller emits the stage tick before AcquireLock.
	spec, err := m.LlamaServerForCurrentPlatform()
	if err != nil {
		return "", nil, err
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
			return target, spec, nil
		}
		logger.Info("ensure llama_server: marker matches but target missing, redoing install",
			"target", target)
	}

	progress := makeProgress(logger, reporter, "llama_server", spec.Size)
	switch spec.Extract {
	case "":
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			return "", nil, fmt.Errorf("ensure llama_server: mkdir: %w", err)
		}

		logger.Info("ensure llama_server: downloading raw binary", "url", spec.URL)

		if err := downloadWithRetry(ctx, spec.URL, spec.SHA256, spec.Size, target, progress, logger, "llama_server"); err != nil {
			return "", nil, fmt.Errorf("ensure llama_server: download: %w", err)
		}
		if err := os.Chmod(target, 0o755); err != nil {
			return "", nil, fmt.Errorf("ensure llama_server: chmod: %w", err)
		}
	case "tar.gz":
		tarPath := filepath.Join(p.Cache, "llama-server.tar.gz")
		logger.Info("ensure llama_server: downloading tarball",
			"url", spec.URL,
			"cache", tarPath)
		if err := downloadWithRetry(ctx, spec.URL, spec.SHA256, spec.Size, tarPath, progress, logger, "llama_server"); err != nil {
			return "", nil, fmt.Errorf("ensure llama_server: download: %w", err)
		}
		defer os.Remove(tarPath)
		logger.Info("ensure llama_server: extracting tarball", "dest", p.LlamaDir)
		extracted, err := ExtractTarGz(tarPath, p.LlamaDir)
		if err != nil {
			return "", nil, fmt.Errorf("ensure llama_server: extract: %w", err)
		}
		logger.Info("ensure llama_server: extracted", "files", len(extracted))
	default:
		return "", nil, fmt.Errorf("manifest: unsupported extract type %q", spec.Extract)
	}

	if _, err := os.Stat(target); err != nil {
		return "", nil, fmt.Errorf("ensure llama_server: exec missing after install: %s: %w", target, err)
	}
	if err := os.Chmod(target, 0o755); err != nil {
		return "", nil, fmt.Errorf("ensure llama_server: chmod target: %w", err)
	}
	// Marker write is an optimization; tolerate failure (we'll just re-extract next boot).
	_ = os.WriteFile(marker, []byte(spec.SHA256), 0o600)
	logger.Info("ensure llama_server: install complete", "target", target)
	return target, spec, nil
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

// Return model path and manifest spec, downloading artifact if not exist
func ensureModel(ctx context.Context, p *Paths, m *Manifest, variant string, logger *slog.Logger, reporter StatusReporter) (string, ArtifactSpec, error) {
	// Caller emits the stage tick before invoking us.
	spec, err := m.ModelSpec(variant)
	if err != nil {
		return "", ArtifactSpec{}, err
	}
	target := p.ModelPath(variant)
	logger.Info("ensure model: target",
		"variant", variant,
		"target", target,
		"want_sha256", spec.SHA256,
		"size", spec.Size)

	ok, err := FileMatchesSHA256(target, spec.SHA256)
	if err != nil {
		return "", ArtifactSpec{}, fmt.Errorf("ensure model: hash existing: %w", err)
	}
	if ok {
		logger.Info("ensure model: cache hit, skipping download")
		return target, spec, nil
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return "", ArtifactSpec{}, err
	}

	logger.Info("ensure model: downloading GGUF", "url", spec.URL)
	progress := makeProgress(logger, reporter, "model", spec.Size)
	if err := downloadWithRetry(ctx, spec.URL, spec.SHA256, spec.Size, target, progress, logger, "model"); err != nil {
		return "", ArtifactSpec{}, fmt.Errorf("ensure model: download: %w", err)
	}
	logger.Info("ensure model: install complete", "target", target)

	return target, spec, nil
}
