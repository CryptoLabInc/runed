package ipc

import (
	"crypto/sha256"
	"encoding/hex"
	"path/filepath"
)

// sunPathLimit is the conservative cross-platform capacity of
// sockaddr_un.sun_path: 104 bytes on macOS, 108 on Linux. A path at or over
// the macOS limit fails bind/connect with EINVAL on at least one supported
// platform, so we treat >= 104 as unusable everywhere.
const sunPathLimit = 104

// ResolveSocketPath returns a bindable/dialable path for the canonical socket
// path (INST-7). Short paths are returned unchanged; a path at/over the
// platform's sun_path limit — e.g. $RUNED_HOME/embedding.sock under a deep
// sandbox HOME — is mapped to a short, DETERMINISTIC alias derived from the
// canonical one, so the daemon (bind) and the client (dial) converge on the
// same path without any handshake.
//
// The alias is /tmp/runed-<hash>.sock where <hash> is the first 8 bytes of
// sha256(canonical) in hex (~32 bytes total, well under the limit). The hash
// covers the full canonical path — which includes the user's home — so
// distinct RUNED_HOMEs and distinct users do not collide. /tmp is joined
// literally rather than via os.TempDir(): on macOS the latter is itself a
// long /var/folders/... path that can re-trip the very limit we're avoiding.
// The input is filepath.Clean-ed before hashing so that differently spelled
// forms of the same path (e.g. an uncleaned WithSocketPath value vs. the
// daemon's filepath.Join of $RUNED_HOME) still hash to the same alias. The
// function is idempotent — an already-resolved alias is short and comes back
// unchanged — so callers may resolve defensively at every bind/dial site.
func ResolveSocketPath(canonical string) string {
	cleaned := filepath.Clean(canonical)
	if len(cleaned) < sunPathLimit {
		return cleaned
	}
	sum := sha256.Sum256([]byte(cleaned))
	return "/tmp/runed-" + hex.EncodeToString(sum[:8]) + ".sock"
}
