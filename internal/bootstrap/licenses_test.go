package bootstrap

import (
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	runed "github.com/CryptoLabInc/runed"
)

// installLicenses must land every embedded license text under
// $RUNED_HOME/licenses/, byte-identical to the embedded copy.
func TestInstallLicenses_WritesAllTextsVerbatim(t *testing.T) {
	home := t.TempDir()
	p := &Paths{Home: home}

	if err := installLicenses(p); err != nil {
		t.Fatalf("installLicenses: %v", err)
	}

	for _, path := range []string{
		"LICENSE",
		"THIRD_PARTY_LICENSES/README.md",
		"THIRD_PARTY_LICENSES/llama.cpp.LICENSE",
		"THIRD_PARTY_LICENSES/Qwen3-Embedding.Apache-2.0.LICENSE",
	} {
		want, err := runed.LicenseFS.ReadFile(path)
		if err != nil {
			t.Fatalf("embedded %s missing: %v", path, err)
		}
		got, err := os.ReadFile(filepath.Join(home, "licenses", path))
		if err != nil {
			t.Fatalf("installed %s missing: %v", path, err)
		}
		if string(got) != string(want) {
			t.Errorf("%s: installed copy differs from embedded text", path)
		}
	}
}

// Every public install entry point must stop before installing an artifact when
// the accompanying license texts cannot be written.
func TestEnsureEntryPointsFailWhenLicensesCannotBeInstalled(t *testing.T) {
	tests := []struct {
		name string
		run  func(*Paths, *Manifest) error
	}{
		{
			name: "all",
			run: func(p *Paths, m *Manifest) error {
				_, _, _, err := EnsureAll(t.Context(), p, m, slog.Default(), nil)
				return err
			},
		},
		{
			name: "llama server",
			run: func(p *Paths, m *Manifest) error {
				_, err := EnsureLlamaServer(t.Context(), p, m, slog.Default(), nil)
				return err
			},
		},
		{
			name: "model",
			run: func(p *Paths, m *Manifest) error {
				_, _, err := EnsureModel(t.Context(), p, m, slog.Default(), nil)
				return err
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			t.Setenv(EnvHome, home)
			p, err := Resolve()
			if err != nil {
				t.Fatalf("resolve paths: %v", err)
			}
			// A regular file blocks creation of the required licenses directory.
			if err := os.WriteFile(filepath.Join(home, "licenses"), []byte("blocked"), 0o600); err != nil {
				t.Fatalf("block license directory: %v", err)
			}
			m := &Manifest{DefaultModel: "test-model"}
			err = tc.run(p, m)
			if err == nil || !strings.Contains(err.Error(), "install licenses") {
				t.Fatalf("error = %v, want install licenses failure", err)
			}
		})
	}
}

// A second run must overwrite cleanly (bootstrap runs on every launch).
func TestInstallLicenses_Idempotent(t *testing.T) {
	home := t.TempDir()
	p := &Paths{Home: home}
	if err := installLicenses(p); err != nil {
		t.Fatalf("first install: %v", err)
	}
	// Corrupt one file; the rerun must restore the embedded text.
	target := filepath.Join(home, "licenses", "LICENSE")
	if err := os.WriteFile(target, []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := installLicenses(p); err != nil {
		t.Fatalf("second install: %v", err)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := runed.LicenseFS.ReadFile("LICENSE")
	if string(got) != string(want) {
		t.Error("rerun did not restore the embedded LICENSE text")
	}
}
