//go:build !windows

package ipc

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveSocketPath_ShortPathUnchanged(t *testing.T) {
	p := "/tmp/runed-test/embedding.sock"
	if got := ResolveSocketPath(p); got != p {
		t.Fatalf("short path changed: got %q want %q", got, p)
	}
}

func TestResolveSocketPath_LongPathAliased(t *testing.T) {
	long := "/" + strings.Repeat("d", 120) + "/embedding.sock"
	if len(long) < sunPathLimit {
		t.Fatalf("test setup: path not long enough (%d bytes)", len(long))
	}
	got := ResolveSocketPath(long)
	wantDir := filepath.Join("/tmp", fmt.Sprintf("runed-%d", os.Getuid()))
	if filepath.Dir(got) != wantDir || !strings.HasPrefix(filepath.Base(got), "runed-") || !strings.HasSuffix(got, ".sock") {
		t.Fatalf("alias has wrong shape: %q", got)
	}
	if len(got) >= sunPathLimit {
		t.Fatalf("alias too long (%d bytes): %q", len(got), got)
	}
	// Deterministic: the same canonical input yields the same alias — this is
	// what lets the daemon and the client converge without a handshake.
	if again := ResolveSocketPath(long); again != got {
		t.Fatalf("not deterministic: %q vs %q", got, again)
	}
	// Distinct canonical inputs (different RUNED_HOMEs/users) must not collide.
	other := "/" + strings.Repeat("e", 120) + "/embedding.sock"
	if ResolveSocketPath(other) == got {
		t.Fatalf("distinct canonical paths mapped to the same alias %q", got)
	}
	// Idempotent: resolving an already-resolved alias returns it unchanged, so
	// defensive double-resolution at bind/dial sites is harmless.
	if re := ResolveSocketPath(got); re != got {
		t.Fatalf("alias not idempotent: %q -> %q", got, re)
	}
}

func TestResolveSocketPath_RelativeAndAbsoluteSpellingsConverge(t *testing.T) {
	relative := filepath.Join(strings.Repeat("d", 60), strings.Repeat("e", 60), "embedding.sock")
	absolute, err := filepath.Abs(relative)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	if ResolveSocketPath(relative) != ResolveSocketPath(absolute) {
		t.Fatalf("relative and absolute spellings diverged: %q vs %q",
			ResolveSocketPath(relative), ResolveSocketPath(absolute))
	}
}

func TestResolveSocketPath_UncleanedSpellingsConverge(t *testing.T) {
	// The client may get an uncleaned path via WithSocketPath while the
	// spawned daemon recomputes it with filepath.Join (which cleans); both
	// spellings must hash to the same alias.
	clean := "/" + strings.Repeat("d", 120) + "/embedding.sock"
	unclean := "/" + strings.Repeat("d", 120) + "//./embedding.sock"
	if ResolveSocketPath(clean) != ResolveSocketPath(unclean) {
		t.Fatalf("cleaned and uncleaned spellings diverged: %q vs %q",
			ResolveSocketPath(clean), ResolveSocketPath(unclean))
	}
}

// TestListen_LongPathBindsAndDials verifies that a canonical socket path over
// the sun_path limit yields a listener reachable through the deterministic
// short alias.
func TestListen_LongPathBindsAndDials(t *testing.T) {
	base := shortTempDir(t)
	deep := filepath.Join(base, strings.Repeat("d", 60), strings.Repeat("e", 60))
	if err := os.MkdirAll(deep, 0o700); err != nil {
		t.Skipf("cannot create deep dir in this sandbox: %v", err)
	}
	canonical := filepath.Join(deep, "embedding.sock")
	if len(canonical) < sunPathLimit {
		t.Fatalf("test setup: canonical path not long enough (%d bytes)", len(canonical))
	}

	// Daemon side: binding the canonical path used to fail with
	// "bind: invalid argument"; Listen now binds the alias.
	lis, err := Listen(canonical)
	if err != nil {
		t.Fatalf("Listen on long canonical path: %v", err)
	}
	defer lis.Close()

	resolved := ResolveSocketPath(canonical)
	if resolved == canonical {
		t.Fatal("long canonical path was not aliased")
	}
	// Client side: dialing the independently resolved alias reaches the
	// daemon's listener.
	conn, err := net.Dial("unix", resolved)
	if err != nil {
		t.Fatalf("dial resolved alias %s: %v", resolved, err)
	}
	conn.Close()

	// The /tmp alias keeps the socket owner-only.
	info, err := os.Stat(resolved)
	if err != nil {
		t.Fatalf("stat alias: %v", err)
	}
	if info.Mode()&0o777 != 0o700 {
		t.Fatalf("want mode 0700 on alias, got %o", info.Mode()&0o777)
	}
	parent, err := os.Stat(filepath.Dir(resolved))
	if err != nil {
		t.Fatalf("stat alias parent: %v", err)
	}
	if parent.Mode().Perm() != 0o700 {
		t.Fatalf("want alias parent mode 0700, got %o", parent.Mode().Perm())
	}
}
