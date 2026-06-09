//go:build windows

package ipc

import (
	"errors"
)

// Listen is not implemented on Windows in Plan A. Named-pipe support lands
// in Plan B.
func Listen(path string) (Listener, error) {
	_ = path
	return nil, errors.New("runed: Windows named pipe not implemented in Plan A")
}
