//go:build !windows

package bootstrap

import (
	"fmt"
	"os"
	"strconv"
	"strings"
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
//
// On success the holder's PID is written into the file as a diagnostic
// hint for trailers that subsequently time out. The PID write is
// best-effort — a missing or unreadable PID just falls back to a generic
// "another instance" message in the timeout error.
func AcquireLock(path string, timeout time.Duration) (*FileLock, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open lock %s: %w", path, err)
	}
	deadline := time.Now().Add(timeout)
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			recordHolderPID(f)
			return &FileLock{f: f}, nil
		}
		if !isWouldBlock(err) {
			f.Close()
			return nil, fmt.Errorf("flock %s: %w", path, err)
		}
		if time.Now().After(deadline) {
			holder := readHolderPID(path)
			f.Close()
			if holder != "" {
				return nil, fmt.Errorf("flock %s: timeout after %s; another runed instance (pid %s) is bootstrapping — retry shortly",
					path, timeout, holder)
			}
			return nil, fmt.Errorf("flock %s: timeout after %s; another runed instance is bootstrapping — retry shortly",
				path, timeout)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// recordHolderPID writes the current process's PID into the locked file.
// All errors are tolerated: a trailer that can't read the PID falls back
// to a generic timeout message and the lock itself is unaffected.
func recordHolderPID(f *os.File) {
	_ = f.Truncate(0)
	_, _ = f.Seek(0, 0)
	_, _ = fmt.Fprintf(f, "%d", os.Getpid())
	_ = f.Sync()
}

// readHolderPID returns the PID written into the lock file by the leader,
// or "" if the file is empty, unreadable, or doesn't contain a valid
// integer (e.g. partially written by a leader that's still racing the
// trailer's read).
func readHolderPID(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	s := strings.TrimSpace(string(data))
	if s == "" {
		return ""
	}
	if _, err := strconv.Atoi(s); err != nil {
		return ""
	}
	return s
}

func isWouldBlock(err error) bool {
	return err == syscall.EWOULDBLOCK || err == syscall.EAGAIN
}

func (l *FileLock) Release() {
	_ = syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
	_ = l.f.Close()
}
