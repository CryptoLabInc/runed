//go:build windows

package backend

// Orphan sweeping is a unix concern in Plan A: the pidfile mechanism guards
// against a SIGKILLed runed leaving its llama-server child behind, and the
// identity check shells out to ps. On Windows the child is tied to the job
// object / console lifetime instead; these are no-ops so the backend builds.

func writeChildPid(path string, pid int, binary string) error { return nil }

func clearChildPid(path string) {}

func sweepOrphanChild(path, binary string) {}
