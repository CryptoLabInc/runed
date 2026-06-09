package ipc

import "net"

// Listener is the net.Listener returned by Listen. Beyond accepting
// connections it can report whether the on-disk socket file still refers to
// the socket it bound. runed uses StillOwned to self-evict when its socket
// path is unlinked or taken over by another daemon — which happens when
// $RUNED_HOME is recreated out from under a running daemon, leaving the old
// process alive but unreachable on a path that now points elsewhere.
type Listener interface {
	net.Listener
	// StillOwned reports whether the socket file at the bind path still
	// refers to this listener's socket. It returns false once the file is
	// removed or replaced by a different socket (different device/inode).
	StillOwned() bool
}
