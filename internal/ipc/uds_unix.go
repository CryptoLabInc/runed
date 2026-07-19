//go:build !windows

// Package ipc provides the local inter-process transport used by the runed
// daemon. On macOS/Linux this is a UNIX domain socket; on Windows (Plan B)
// it will be a named pipe.
package ipc

import (
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
	"syscall"
)

// Listen binds a unix domain socket at path with 0700 permissions.
//
// The parent directory is created (0700) if missing. Stale socket files
// (leftover from a crashed daemon) are unlinked and re-bound. If another
// process is actively listening on the same path, Listen returns an error.
//
// The returned Listener disables Go's default unlink-on-close and instead
// removes the socket file only if it still refers to the socket this call
// bound (see ownedListener.Close). Go's default would unlink whatever file
// occupies the path on Close, even one a *different* daemon rebound there —
// which can delete a healthy daemon's socket after $RUNED_HOME is recreated
// out from under a running daemon. StillOwned exposes the same identity check
// so the daemon can self-evict when its socket is taken over.
//
// A path at/over the sun_path limit is transparently remapped via
// ResolveSocketPath before binding; binding the canonical path
// would fail with a cryptic "bind: invalid argument". Clients resolve the
// same canonical path before dialing, so both sides meet at the alias.
func Listen(path string) (Listener, error) {
	path = ResolveSocketPath(path)
	if err := ensureSocketParent(path); err != nil {
		return nil, fmt.Errorf("mkdir parent: %w", err)
	}

	if info, err := os.Stat(path); err == nil {
		if info.Mode()&os.ModeSocket == 0 {
			// Not a socket — be conservative, remove so Listen can bind.
			if err := os.Remove(path); err != nil {
				return nil, fmt.Errorf("remove non-socket at %s: %w", path, err)
			}
		} else {
			// Existing socket — probe for live listener.
			if conn, derr := net.Dial("unix", path); derr == nil {
				conn.Close()
				return nil, fmt.Errorf("socket %s already in use", path)
			}
			// Stale: no listener. Remove and rebind below.
			_ = os.Remove(path)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}

	lis, err := net.Listen("unix", path)
	if err != nil {
		return nil, fmt.Errorf("listen: %w", err)
	}
	ul, ok := lis.(*net.UnixListener)
	if !ok {
		// "unix" always yields *net.UnixListener; defensive. unlinkOnClose is
		// still at its default (true) here, so Close cleans up the file.
		_ = lis.Close()
		return nil, fmt.Errorf("listen: unexpected listener type %T", lis)
	}
	// Take ownership of unlinking so a different daemon's socket at this path
	// is never deleted by our Close (see ownedListener.Close).
	ul.SetUnlinkOnClose(false)

	if err := os.Chmod(path, 0o700); err != nil {
		_ = ul.Close()
		_ = os.Remove(path)
		return nil, fmt.Errorf("chmod: %w", err)
	}
	dev, ino, ok := socketIdentity(path)
	if !ok {
		_ = ul.Close()
		_ = os.Remove(path)
		return nil, fmt.Errorf("stat bound socket %s", path)
	}
	return &ownedListener{UnixListener: ul, path: path, dev: dev, ino: ino}, nil
}

// ownedListener is a *net.UnixListener that remembers the identity (device +
// inode) of the socket file it bound, so it can tell whether the file at its
// path is still its own socket and avoid unlinking one a different daemon
// rebound at the same path.
type ownedListener struct {
	*net.UnixListener
	path string
	dev  uint64
	ino  uint64
	once sync.Once
}

// Close stops accepting and removes the socket file, but only if the file
// still refers to this listener's socket. SetUnlinkOnClose(false) in Listen
// disabled net's unconditional unlink, so this is the sole remover — and it
// will not delete a socket another daemon rebound at the same path.
func (l *ownedListener) Close() error {
	err := l.UnixListener.Close()
	l.once.Do(func() {
		if sameSocket(l.path, l.dev, l.ino) {
			_ = os.Remove(l.path)
		}
	})
	return err
}

// StillOwned reports whether the socket file at the bind path still refers to
// the socket this listener bound. It returns false if the file was removed or
// replaced by a different socket — e.g. after $RUNED_HOME was wiped and
// another daemon rebound the path.
func (l *ownedListener) StillOwned() bool {
	return sameSocket(l.path, l.dev, l.ino)
}

// socketIdentity returns the device and inode of the file at path.
func socketIdentity(path string) (dev, ino uint64, ok bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, 0, false
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, 0, false
	}
	// Dev is int32 on darwin and uint64 on linux; widen uniformly.
	return uint64(st.Dev), uint64(st.Ino), true
}

// sameSocket reports whether the file at path currently has the given device
// and inode. A live listener pins its own inode, so that inode cannot be
// reused for a different file while we are alive — making (dev,ino) a stable
// identity for "is the path still my socket".
func sameSocket(path string, dev, ino uint64) bool {
	d, i, ok := socketIdentity(path)
	return ok && d == dev && i == ino
}
