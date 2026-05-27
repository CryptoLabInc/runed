package bootstrap

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModelVariant_EnvWins(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "from-env")

	p, _ := Resolve()
	// Even with config + manifest defaults set, env wins.
	writeConfig(t, p.Config, `{"version":1,"model_variant":"from-config"}`)
	m := &Manifest{DefaultModel: "from-manifest"}

	got, err := ResolveModelVariant(p, m)
	if err != nil {
		t.Fatalf("ResolveModelVariant: %v", err)
	}
	if got != "from-env" {
		t.Errorf("got %q, want from-env", got)
	}
}

func TestResolveModelVariant_ConfigOverManifest(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")

	p, _ := Resolve()
	writeConfig(t, p.Config, `{"version":1,"model_variant":"from-config"}`)
	m := &Manifest{DefaultModel: "from-manifest"}

	got, err := ResolveModelVariant(p, m)
	if err != nil {
		t.Fatalf("ResolveModelVariant: %v", err)
	}
	if got != "from-config" {
		t.Errorf("got %q, want from-config", got)
	}
}

func TestResolveModelVariant_ManifestDefault(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")

	p, _ := Resolve()
	// No config file written.
	m := &Manifest{DefaultModel: "from-manifest"}

	got, err := ResolveModelVariant(p, m)
	if err != nil {
		t.Fatalf("ResolveModelVariant: %v", err)
	}
	if got != "from-manifest" {
		t.Errorf("got %q, want from-manifest", got)
	}
}

func TestResolveModelVariant_NoneSpecified(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	t.Setenv(EnvModelVariant, "")

	p, _ := Resolve()
	m := &Manifest{}
	if _, err := ResolveModelVariant(p, m); err == nil {
		t.Fatal("expected error when no variant source set")
	}
}

func TestLlamaServerTarget_DefaultExec(t *testing.T) {
	p := &Paths{LlamaDir: "/x/bin/llama-cpp"}
	got := llamaServerTarget(p, &LlamaServerSpec{}) // Exec unset
	want := filepath.Join("/x/bin/llama-cpp", "llama-server")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestLlamaServerTarget_NestedExec(t *testing.T) {
	p := &Paths{LlamaDir: "/x/bin/llama-cpp"}
	got := llamaServerTarget(p, &LlamaServerSpec{Exec: "build/bin/llama-server"})
	want := filepath.Join("/x/bin/llama-cpp", "build", "bin", "llama-server")
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func writeConfig(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
}
