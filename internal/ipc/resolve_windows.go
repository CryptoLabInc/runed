//go:build windows

package ipc

import "path/filepath"

// Windows support uses named pipes rather than sockaddr_un, so the Unix
// short-path aliasing rule does not apply.
func ResolveSocketPath(path string) string { return filepath.Clean(path) }
