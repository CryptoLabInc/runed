//go:build !windows

package backend

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// sleepBin returns an absolute path to the system sleep binary so argv[0]
// comparisons are exact.
func sleepBin(t *testing.T) string {
	t.Helper()
	p, err := exec.LookPath("sleep")
	if err != nil {
		t.Skipf("no sleep binary: %v", err)
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	return abs
}

func waitGone(t *testing.T, pid int, within time.Duration) {
	t.Helper()
	deadline := time.Now().Add(within)
	for syscall.Kill(pid, 0) == nil {
		if time.Now().After(deadline) {
			t.Fatalf("pid %d still alive after %v", pid, within)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// TestSweepReapsMatchingOrphan is the scenario the pidfile exists for: a
// llama-server (stand-in: sleep) whose runed died by SIGKILL is still alive
// at the next boot, its pid+binary recorded. The sweep must kill it and
// consume the record.
func TestSweepReapsMatchingOrphan(t *testing.T) {
	bin := sleepBin(t)
	orphan := exec.Command(bin, "60")
	if err := orphan.Start(); err != nil {
		t.Fatalf("spawn orphan: %v", err)
	}
	// Reap in the background so the killed child doesn't linger as a zombie
	// (a zombie still shows in ps and would confuse waitGone).
	go func() { _ = orphan.Wait() }()
	defer func() { _ = orphan.Process.Kill() }()

	pidPath := filepath.Join(t.TempDir(), "llama.pid")
	if err := writeChildPid(pidPath, orphan.Process.Pid, bin); err != nil {
		t.Fatalf("writeChildPid: %v", err)
	}

	sweepOrphanChild(pidPath, bin)

	waitGone(t, orphan.Process.Pid, 5*time.Second)
	if _, err := os.Stat(pidPath); !os.IsNotExist(err) {
		t.Fatalf("pidfile not consumed after sweep: %v", err)
	}
}

// TestSweepSparesReusedPid: the recorded pid now belongs to a different
// process (here: the test binary itself). The sweep must not kill it and
// must clear the stale record.
func TestSweepSparesReusedPid(t *testing.T) {
	bin := sleepBin(t)
	pidPath := filepath.Join(t.TempDir(), "llama.pid")
	if err := writeChildPid(pidPath, os.Getpid(), bin); err != nil {
		t.Fatalf("writeChildPid: %v", err)
	}

	sweepOrphanChild(pidPath, bin)

	if syscall.Kill(os.Getpid(), 0) != nil {
		t.Fatal("sweep killed an innocent process on pid reuse")
	}
	if _, err := os.Stat(pidPath); !os.IsNotExist(err) {
		t.Fatalf("stale pidfile not cleared: %v", err)
	}
}

// TestSweepIgnoresMissingRecord: the common boot (clean previous shutdown)
// has no pidfile and the sweep must be a silent no-op.
func TestSweepIgnoresMissingRecord(t *testing.T) {
	sweepOrphanChild(filepath.Join(t.TempDir(), "llama.pid"), "/nonexistent")
}

// TestWatchChildClearsPidfile: the record exists exactly while a child runs —
// once watchChild reaps the child, the file must be gone (so a later boot
// doesn't sweep a pid that already exited).
func TestWatchChildClearsPidfile(t *testing.T) {
	pidPath := filepath.Join(t.TempDir(), "llama.pid")
	b := NewLlamaBackend(Config{PidPath: pidPath})

	cmd := attachPlaceholderChild(t, b, 1)
	if err := writeChildPid(pidPath, cmd.Process.Pid, "placeholder"); err != nil {
		t.Fatalf("writeChildPid: %v", err)
	}

	b.mu.Lock()
	done := b.cmdDone
	b.mu.Unlock()
	_ = cmd.Process.Kill()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("watchChild never reaped the child")
	}

	if _, err := os.Stat(pidPath); !os.IsNotExist(err) {
		t.Fatalf("pidfile survived the child reap: %v", err)
	}
}
