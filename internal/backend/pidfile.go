//go:build !windows

package backend

import (
	"encoding/json"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// The llama-server child outlives runed when runed dies by SIGKILL (force
// quit, crash, memory-pressure jetsam): SIGKILL cannot be intercepted, and on
// macOS there is no Pdeathsig, so the graceful-shutdown path never runs and
// the child is reparented to pid 1 holding ~1.7 GB. The next runed then
// spawns a second llama-server — doubling memory, silently, since the fresh
// daemon works fine. Reclaiming at death is impossible; reclaiming at the
// next boot is not: runed records its child in a pidfile and Start sweeps a
// matching orphan before spawning.

// childRecord is the persisted identity of the spawned llama-server. Binary
// is the fingerprint: pid reuse across reboots (or by unrelated processes)
// must never kill an innocent process, so the sweep only acts when the live
// process's argv[0] still equals the llama-server binary this runed launches.
type childRecord struct {
	PID    int    `json:"pid"`
	Binary string `json:"binary"`
}

// writeChildPid records the spawned child atomically (tmp + rename), so a
// crash mid-write leaves either the old record or the new one, never a
// truncated file. Best-effort: a failure only degrades orphan sweeping, not
// serving, so the caller logs and continues.
func writeChildPid(path string, pid int, binary string) error {
	if path == "" {
		return nil
	}
	data, err := json.Marshal(childRecord{PID: pid, Binary: binary})
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".llama.pid.*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // no-op after successful rename
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// clearChildPid removes the record once the child has been reaped — the
// pidfile exists exactly while a child is (believed) running, so a surviving
// file at boot means a possible orphan.
func clearChildPid(path string) {
	if path != "" {
		_ = os.Remove(path)
	}
}

// processRunsBinary reports whether pid is alive and executing binary, by
// matching `ps -o args=` (full argv; `comm=` would be truncated to 15 chars
// on linux) against the recorded binary path. Prefix matching handles binary
// paths containing spaces (e.g. under "Application Support"): argv must be
// exactly the binary or the binary followed by its arguments.
func processRunsBinary(pid int, binary string) bool {
	out, err := exec.Command("ps", "-o", "args=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return false // ps exits non-zero for a nonexistent pid
	}
	args := strings.TrimSpace(string(out))
	return args == binary || strings.HasPrefix(args, binary+" ")
}

// sweepOrphanChild kills a llama-server left behind by a previous runed that
// died without cleanup (SIGKILL/jetsam). It only acts when the recorded pid
// is alive AND still runs the recorded binary — the identity we captured at
// spawn time, so an upgraded runed (new binary path) still reaps the old
// version's orphan, while a reused pid or unreadable record only clears the
// file. Called from Start before the first spawn; a missing pidfile is the
// common case (clean previous shutdown) and a no-op.
func sweepOrphanChild(path, binary string) {
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return // no record — nothing was running (or nothing to know)
	}
	// The record is consumed either way: after this boot the only child we
	// are responsible for is the one we spawn next.
	defer clearChildPid(path)

	var rec childRecord
	if err := json.Unmarshal(data, &rec); err != nil || rec.PID <= 0 || rec.Binary == "" {
		slog.Warn("backend: unreadable llama pidfile, ignoring", "path", path)
		return
	}
	if !processRunsBinary(rec.PID, rec.Binary) {
		// Exited with its parent, or the pid was reused by an unrelated
		// process — never kill on a stale identity.
		return
	}
	slog.Warn("backend: reaping orphaned llama-server from a previous runed",
		"pid", rec.PID, "binary", rec.Binary, "spawning", binary)
	_ = syscall.Kill(rec.PID, syscall.SIGTERM)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if syscall.Kill(rec.PID, 0) != nil {
			return // exited
		}
		time.Sleep(100 * time.Millisecond)
	}
	_ = syscall.Kill(rec.PID, syscall.SIGKILL)
}

