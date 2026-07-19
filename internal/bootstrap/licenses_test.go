package bootstrap

import (
	"os"
	"path/filepath"
	"testing"

	runed "github.com/CryptoLabInc/runed"
)

// installLicenses must land every embedded license text under
// $RUNED_HOME/licenses/, byte-identical to the embedded copy (OPS-90).
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
