package bootstrap

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	AuditArtifactLlamaServer = "llama_server"
	AuditArtifactModel       = "model"
)

type InstalledManifest struct {
	ManifestURL     string                       `json:"manifest_url"`
	ManifestVersion int                          `json:"manifest_version"`
	Platform        string                       `json:"platform"`
	InstalledAt     string                       `json:"installed_at"` // UTC RFC3339 timestamp
	ModelVariant    string                       `json:"model_variant,omitempty"`
	Artifacts       map[string]InstalledArtifact `json:"artifacts"`
}

type InstalledArtifact struct {
	// Tarbar info
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`

	// Extracted binary (e.g., llama-server)
	Path string `json:"path"` // SHA of this file may not matched with SHA of URL
	Size int64  `json:"size,omitempty"`
}

func InstalledManifestPath(p *Paths) string {
	return filepath.Join(p.Home, "installed.json")
}

func WriteInstalledManifest(p *Paths, im *InstalledManifest) error {
	data, err := json.MarshalIndent(im, "", "  ")
	if err != nil {
		return fmt.Errorf("installed.json: marshal: %w", err)
	}

	target := InstalledManifestPath(p)
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return fmt.Errorf("installed.json: mkdir parent: %w", err)
	}

	// Atomic write by writing to tmp then renaming
	tmp, err := os.CreateTemp(filepath.Dir(target), ".installed.json.*")
	if err != nil {
		return fmt.Errorf("installed.json: tmp: %w", err)
	}

	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("installed.json: write: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("installed.json: close: %w", err)
	}
	if err := os.Chmod(tmpName, 0o600); err != nil {
		return fmt.Errorf("installed.json: chmod: %w", err)
	}
	if err := os.Rename(tmpName, target); err != nil {
		return fmt.Errorf("installed.json: rename: %w", err)
	}

	return nil
}

func ReadInstalledManifest(p *Paths) (*InstalledManifest, error) {
	data, err := os.ReadFile(InstalledManifestPath(p))
	if err != nil {
		return nil, err
	}

	var record InstalledManifest
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("installed.json: parse: %w", err)
	}

	return &record, nil
}

func recordInstall(p *Paths, manifestURL string, manifestVersion int, modelVariant string, artifacts map[string]InstalledArtifact) error {
	cur, err := ReadInstalledManifest(p)
	if err != nil && !os.IsNotExist(err) {
		cur = nil // use fresh data on failure
	}
	if cur == nil {
		cur = &InstalledManifest{}
	}

	if cur.Artifacts == nil {
		cur.Artifacts = map[string]InstalledArtifact{}
	}

	cur.ManifestURL = manifestURL
	cur.ManifestVersion = manifestVersion
	cur.Platform = PlatformTuple()
	cur.InstalledAt = time.Now().UTC().Format(time.RFC3339)
	if modelVariant != "" {
		cur.ModelVariant = modelVariant
	}
	for k, v := range artifacts {
		cur.Artifacts[k] = v
	}

	return WriteInstalledManifest(p, cur)
}

func statSize(path string, fallback int64) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return fallback
	}

	return info.Size()
}
