package bootstrap

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolve_HomeEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)

	p, err := Resolve()
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Home != dir {
		t.Errorf("Home: got %q, want %q", p.Home, dir)
	}
	if p.Bin != filepath.Join(dir, "bin") {
		t.Errorf("Bin: got %q", p.Bin)
	}
	if p.LlamaDir != filepath.Join(dir, "bin", "llama-cpp") {
		t.Errorf("LlamaDir: got %q", p.LlamaDir)
	}
	if p.Models != filepath.Join(dir, "models") {
		t.Errorf("Models: got %q", p.Models)
	}
	if p.Config != filepath.Join(dir, "config.json") {
		t.Errorf("Config: got %q", p.Config)
	}
	if p.InstallLock != filepath.Join(dir, "install.lock") {
		t.Errorf("InstallLock: got %q", p.InstallLock)
	}
}

func TestResolve_DefaultHome(t *testing.T) {
	t.Setenv(EnvHome, "")
	u, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no user home: %v", err)
	}
	p, err := Resolve()
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	want := filepath.Join(u, ".runed")
	if p.Home != want {
		t.Errorf("Home: got %q, want %q", p.Home, want)
	}
}

func TestEnsureDirs(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvHome, dir)
	p, err := Resolve()
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if err := p.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	for _, d := range []string{p.Bin, p.LlamaDir, p.Models, p.Logs, p.Cache} {
		info, err := os.Stat(d)
		if err != nil {
			t.Errorf("missing dir %q: %v", d, err)
			continue
		}
		if !info.IsDir() {
			t.Errorf("%q is not a directory", d)
		}
	}
}

func TestModelPath(t *testing.T) {
	p := &Paths{Models: "/x/models"}
	got := p.ModelPath("qwen3-embedding-0.6b.q6_K")
	want := "/x/models/qwen3-embedding-0.6b.q6_K.gguf"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}
