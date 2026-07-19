//go:build !windows

package ipc

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
)

// sunPathLimit is the conservative cross-platform capacity of
// sockaddr_un.sun_path: 104 bytes on macOS, 108 on Linux. A path at or over
// the macOS limit fails bind/connect with EINVAL on at least one supported
// platform, so we treat >= 104 as unusable everywhere.
const sunPathLimit = 104

// socketAliasDir is deliberately short and private to the current uid. Putting
// the socket directly in world-writable /tmp would let another local user bind
// the predictable alias first and impersonate or deny the daemon.
func socketAliasDir() string {
	return filepath.Join("/tmp", fmt.Sprintf("runed-%d", os.Getuid()))
}

// ResolveSocketPath returns a bindable/dialable path for the canonical socket
// path (INST-7). Short absolute paths are returned unchanged; a path at/over the
// sun_path limit is mapped to a deterministic alias under a short, per-user
// runtime directory so daemon and client converge without a handshake.
//
// Inputs are made absolute before both the length check and hash. In particular,
// spawn may receive a relative WithSocketPath while runed resolves RUNED_HOME to
// an absolute path; hashing their original spellings would make the two sides
// choose different aliases. The alias is idempotent because it is already an
// absolute path shorter than sunPathLimit.
func ResolveSocketPath(canonical string) string {
	cleaned := filepath.Clean(canonical)
	if absolute, err := filepath.Abs(cleaned); err == nil {
		cleaned = absolute
	}
	if len(cleaned) < sunPathLimit {
		return cleaned
	}
	sum := sha256.Sum256([]byte(cleaned))
	return filepath.Join(socketAliasDir(), "runed-"+hex.EncodeToString(sum[:16])+".sock")
}

// ensureSocketParent creates the socket's parent and, for a short-path alias,
// verifies that the shared-runtime entry is a real directory owned by this uid
// with no group/other access. An attacker-created symlink or foreign-owned
// directory fails closed instead of moving the unauthenticated local gRPC
// endpoint into an untrusted namespace.
func ensureSocketParent(path string) error {
	dir := filepath.Dir(path)
	if dir != socketAliasDir() {
		return os.MkdirAll(dir, 0o700)
	}
	if err := os.Mkdir(dir, 0o700); err != nil && !os.IsExist(err) {
		return err
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return fmt.Errorf("socket alias parent %s is not a real directory", dir)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || uint64(stat.Uid) != uint64(os.Getuid()) {
		return fmt.Errorf("socket alias parent %s is not owned by the current user", dir)
	}
	if info.Mode().Perm() != 0o700 {
		if err := os.Chmod(dir, 0o700); err != nil {
			return fmt.Errorf("secure socket alias parent %s: %w", dir, err)
		}
	}
	return nil
}
