package spawn

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/CryptoLabInc/runed/internal/ipc"
)

// fakeRunedScript is a shell script that mimics what the real runed binary
// does *just enough* for EnsureDaemon's Health-poll to see it: bind a UDS,
// answer a single byte, then sleep. We don't speak gRPC here, so we use a
// custom probe (see isDaemonReachable in the test).
const fakeRunedScript = `#!/bin/sh
HOME="$RUNED_HOME"
mkdir -p "$HOME"
SOCK="$HOME/embedding.sock"
rm -f "$SOCK"
# Use nc in listen mode to bind the socket. Stay alive for 60s so the test
# can probe it and then clean up.
nc -lkU "$SOCK" >/dev/null 2>&1 &
echo $! > "$HOME/fake.pid"
sleep 60
`

func TestEnsureDaemon_AlreadyRunning(t *testing.T) {
	// If a daemon is reachable, EnsureDaemon must return nil without doing
	// anything else. We simulate this by binding a UDS ourselves before
	// calling EnsureDaemon. Use a short tempdir because macOS's $TMPDIR
	// paths can exceed sockaddr_un's 104-byte limit and fail bind.
	home := shortTempDir(t)
	sock := filepath.Join(home, "embedding.sock")
	stop := bindFakeListener(t, sock)
	defer stop()

	// (Even with no config file, EnsureDaemon should not try to spawn.)
	t.Setenv("RUNED_CONFIG", "/tmp/runed-no-such-file")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	// For this test we override isDaemonReachable via a thin contract:
	// EnsureDaemon should *short-circuit* on a working socket. We assert
	// this by giving it a non-existent config — if it tried to spawn,
	// it'd fail with "config not found".
	if err := EnsureDaemon(ctx, sock); err != nil {
		t.Fatalf("EnsureDaemon: %v", err)
	}
}

// A deep RUNED_HOME must still take the already-running fast path: both the
// listener and spawn probe resolve the same short alias instead of attempting
// to dial the over-limit canonical path.
func TestEnsureDaemon_AlreadyRunningAtLongPath(t *testing.T) {
	home := filepath.Join(shortTempDir(t), strings.Repeat("d", 60), strings.Repeat("e", 60))
	canonical := filepath.Join(home, "embedding.sock")
	lis, err := ipc.Listen(canonical)
	if err != nil {
		t.Fatalf("listen long path: %v", err)
	}
	defer lis.Close()

	t.Setenv("RUNED_CONFIG", "/tmp/runed-no-such-file")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := EnsureDaemon(ctx, canonical); err != nil {
		t.Fatalf("EnsureDaemon long path: %v", err)
	}
}

// bindFakeListener opens a UDS at path that simply accepts connections
// and immediately closes them. Returns a stop func to unbind. We use it
// as a stand-in for "a daemon is reachable".
func bindFakeListener(t *testing.T, path string) func() {
	t.Helper()
	return startFakeListener(t, path)
}

// TestEnsureDaemon_SpawnFlow verifies the full spawn path: no daemon
// running → read config → exec the fake script → wait for socket → return.
func TestEnsureDaemon_SpawnFlow(t *testing.T) {
	if _, err := exec.LookPath("nc"); err != nil {
		t.Skip("nc not available; skipping spawn smoke")
	}
	// Use a short tempdir: macOS's default $TMPDIR can exceed sockaddr_un's
	// 104-byte limit, which would make the fake daemon's nc -lkU fail bind.
	home := shortTempDir(t)
	t.Setenv("RUNED_HOME", home)

	fakeRuned := filepath.Join(home, "fake-runed")
	if err := os.WriteFile(fakeRuned, []byte(fakeRunedScript), 0o755); err != nil {
		t.Fatalf("write fake runed: %v", err)
	}
	// The fake daemon still needs *something* on disk for the validate
	// step in LoadConfig; reuse the helpers from config_test.go.
	llamaBin := writeFile(t, home, "llama-server", []byte("x"), true)
	model := writeValidGGUF(t, home, "model.gguf")
	cfgPath := writeFile(t, home, "config.json", []byte(fmt.Sprintf(`{
        "version": 1,
        "llama_server": %q,
        "model": %q,
        "runed_binary": %q
    }`, llamaBin, model, fakeRuned)), false)
	t.Setenv("RUNED_CONFIG", cfgPath)

	sock := filepath.Join(home, "embedding.sock")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := EnsureDaemon(ctx, sock); err != nil {
		t.Fatalf("EnsureDaemon: %v", err)
	}
	// The fake script writes its child PID — clean it up.
	pidBytes, _ := os.ReadFile(filepath.Join(home, "fake.pid"))
	if len(pidBytes) > 0 {
		exec.Command("kill", string(bytes.TrimSpace(pidBytes))).Run()
	}
}

// TestEnsureDaemon_HostileSocketPath verifies that probeDaemon's hostile
// classification (e.g. a regular file at the socket path → ENOTSOCK)
// surfaces as an error rather than silently triggering a spawn that
// would clobber the existing entry.
func TestEnsureDaemon_HostileSocketPath(t *testing.T) {
	home := shortTempDir(t)
	sock := filepath.Join(home, "embedding.sock")
	// Create a regular file at the socket path — dial will return ENOTSOCK.
	if err := os.WriteFile(sock, []byte("not a socket"), 0o600); err != nil {
		t.Fatalf("seed regular file: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := EnsureDaemon(ctx, sock); err == nil {
		t.Fatal("expected hostile probe error, got nil")
	} else if !strings.Contains(err.Error(), "probe") {
		t.Fatalf("expected error containing \"probe\", got: %v", err)
	}
}
