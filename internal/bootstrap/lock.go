//go:build !windows

package bootstrap

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

// FileLock is an advisory exclusive flock on a path. Use AcquireLock to
// obtain; call Release exactly once.
type FileLock struct {
	f *os.File
}

// AcquireLock opens (or creates) path and tries to obtain an exclusive
// flock, polling every 100ms until either successful or the timeout
// expires. The lock file is left on disk after Release; it's an
// addressable rendezvous point, not state.
func AcquireLock(path string, timeout time.Duration) (*FileLock, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open lock %s: %w", path, err)
	}
	deadline := time.Now().Add(timeout)
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return &FileLock{f: f}, nil
		}
		if !isWouldBlock(err) {
			f.Close()
			return nil, fmt.Errorf("flock %s: %w", path, err)
		}
		if time.Now().After(deadline) {
			f.Close()
			return nil, fmt.Errorf("flock %s: timeout after %s", path, timeout)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func isWouldBlock(err error) bool {
	return err == syscall.EWOULDBLOCK || err == syscall.EAGAIN
}

func (l *FileLock) Release() {
	_ = syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
	_ = l.f.Close()
}
