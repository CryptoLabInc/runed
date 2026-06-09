//go:build !windows

package ipc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// shortTempDir returns a per-test temp dir under /tmp. macOS's $TMPDIR and
// Go's t.TempDir() produce paths that can exceed the 104-byte sockaddr_un
// limit, causing bind EINVAL for unrelated reasons. /tmp keeps paths short.
func shortTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("/tmp", "runed-ipc-")
	if err != nil {
		t.Fatalf("mkdtemp: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

func TestListen_CreatesSocketWith0700(t *testing.T) {
	dir := shortTempDir(t)
	sockPath := filepath.Join(dir, "embedding.sock")

	lis, err := Listen(sockPath)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	defer lis.Close()

	info, err := os.Stat(sockPath)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode()&0o777 != 0o700 {
		t.Fatalf("want mode 0700, got %o", info.Mode()&0o777)
	}
}

func TestListen_CleansUpStaleSocket(t *testing.T) {
	dir := shortTempDir(t)
	sockPath := filepath.Join(dir, "embedding.sock")

	// Create a stale file manually (no listener).
	f, _ := os.Create(sockPath)
	f.Close()

	// Listen should remove and rebind.
	lis, err := Listen(sockPath)
	if err != nil {
		t.Fatalf("Listen over stale: %v", err)
	}
	defer lis.Close()

	if !lis.StillOwned() {
		t.Fatal("freshly bound socket should be owned")
	}
}

func TestListen_RejectsLiveSocket(t *testing.T) {
	dir := shortTempDir(t)
	sockPath := filepath.Join(dir, "embedding.sock")

	l1, err := Listen(sockPath)
	if err != nil {
		t.Fatalf("first Listen: %v", err)
	}
	defer l1.Close()

	_, err = Listen(sockPath)
	if err == nil {
		t.Fatal("expected error on second Listen, got nil")
	}
	if !strings.Contains(err.Error(), "already in use") {
		t.Fatalf("want 'already in use' error, got: %v", err)
	}
}

func TestStillOwned_FalseAfterRemove(t *testing.T) {
	dir := shortTempDir(t)
	sockPath := filepath.Join(dir, "embedding.sock")

	lis, err := Listen(sockPath)
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	defer lis.Close()

	if !lis.StillOwned() {
		t.Fatal("want owned immediately after bind")
	}
	if err := os.Remove(sockPath); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if lis.StillOwned() {
		t.Fatal("want not owned after socket file removed")
	}
}

// TestClose_DoesNotRemoveReboundSocket is the Fix-2 regression: an evicted
// listener closing must not delete the socket file a *different* listener
// rebound at the same path (Go's default unlink-on-close would).
func TestClose_DoesNotRemoveReboundSocket(t *testing.T) {
	dir := shortTempDir(t)
	sockPath := filepath.Join(dir, "embedding.sock")

	l1, err := Listen(sockPath)
	if err != nil {
		t.Fatalf("first Listen: %v", err)
	}

	// Simulate another daemon rebinding the path after l1's home was wiped:
	// remove the path (l1 keeps its fd, so its inode is pinned), then bind a
	// fresh socket with a new inode at the same path.
	if err := os.Remove(sockPath); err != nil {
		t.Fatalf("remove: %v", err)
	}
	l2, err := Listen(sockPath)
	if err != nil {
		t.Fatalf("second Listen: %v", err)
	}
	defer l2.Close()

	if l1.StillOwned() {
		t.Fatal("l1 should no longer own the rebound path")
	}
	if !l2.StillOwned() {
		t.Fatal("l2 should own the path it just bound")
	}

	// Closing the evicted l1 must leave l2's socket file intact.
	_ = l1.Close()

	if _, err := os.Stat(sockPath); err != nil {
		t.Fatalf("socket path should still exist after evicted l1.Close(): %v", err)
	}
	if !l2.StillOwned() {
		t.Fatal("l1.Close() deleted l2's socket file — unlink-on-close regression")
	}
}
