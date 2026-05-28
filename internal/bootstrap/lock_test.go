//go:build !windows

package bootstrap

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Pins the diagnostic side of the lock: the leader's PID lands in the
// lock file, and a trailer that times out gets that PID surfaced in
// the error message instead of a bare flock failure.
func TestAcquireLock_TimeoutSurfacesHolderPID(t *testing.T) {
	lockPath := filepath.Join(t.TempDir(), "install.lock")

	leader, err := AcquireLock(lockPath, time.Second)
	if err != nil {
		t.Fatalf("leader acquire: %v", err)
	}
	defer leader.Release()

	data, err := os.ReadFile(lockPath)
	if err != nil {
		t.Fatalf("read lock file: %v", err)
	}
	want := fmt.Sprintf("%d", os.Getpid())
	if got := strings.TrimSpace(string(data)); got != want {
		t.Errorf("lock file PID: got %q, want %q", got, want)
	}

	// Same-process-different-fd: syscall.Flock returns EWOULDBLOCK on
	// Linux/Darwin in this case, so a short-timeout AcquireLock will
	// take the timeout branch — exactly the trailer path we want to
	// exercise.
	_, err = AcquireLock(lockPath, 150*time.Millisecond)
	if err == nil {
		t.Fatal("trailer: expected timeout error, got nil")
	}
	wantSub := fmt.Sprintf("pid %d", os.Getpid())
	if !strings.Contains(err.Error(), wantSub) {
		t.Errorf("trailer error should mention holder pid; got: %v", err)
	}
}
