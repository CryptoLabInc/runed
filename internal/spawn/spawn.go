//go:build !windows

package spawn

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
)

// dialResult enumerates the meaningful outcomes of probing the daemon
// socket. "reachable" means a listener accepts our dial; "absent" means
// the socket file is missing or refusing connections (the only states
// where auto-spawn is the right response); "hostile" means we saw a
// path that exists but we can't dial it (permission denied, not a
// socket, etc.) — auto-spawn would clobber whatever's there, so we
// surface this as an error to the caller.
type dialResult int

const (
	dialReachable dialResult = iota
	dialAbsent
	dialHostile
)

// EnsureDaemon makes sure a runed daemon is reachable at socketPath. If
// one is already running, returns nil immediately. Otherwise serializes
// with other clients via a flock at $(dirname socketPath)/spawn.lock,
// reads ~/.runed/config.json, execs the runed binary detached, and
// polls Health until ready or a 10-second timeout.
func EnsureDaemon(ctx context.Context, socketPath string) error {
	switch r, err := probeDaemon(ctx, socketPath); r {
	case dialReachable:
		return nil
	case dialHostile:
		return err
	}
	lockPath := filepath.Join(filepath.Dir(socketPath), "spawn.lock")

	// Ensure the parent directory of the lock file exists before opening
	// it; on a fresh install, ~/.runed/ may not yet exist.
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o700); err != nil {
		return fmt.Errorf("mkdir runed home: %w", err)
	}

	lock, err := acquireLock(lockPath, 30*time.Second)
	if err != nil {
		return fmt.Errorf("acquire spawn lock: %w", err)
	}
	// Held through waitForDaemon — keeps the next caller's recheck blocked
	// until our daemon's listen() is up, closing the post-lock race window.
	defer lock.Release()
	// Race-free recheck: someone may have spawned the daemon while we
	// were waiting on the lock.
	switch r, err := probeDaemon(ctx, socketPath); r {
	case dialReachable:
		return nil
	case dialHostile:
		return err
	}
	cfg, err := LoadConfig()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	if err := launchDaemon(cfg, socketPath); err != nil {
		return err
	}
	return waitForDaemon(ctx, socketPath, 10*time.Second)
}

// probeDaemon dials the socket with a short timeout and classifies the
// outcome. Used in place of a boolean reachability check so that hostile
// states (mode mismatch, non-socket regular file at the path) don't
// silently trigger an auto-spawn that would clobber the other process.
// We don't speak gRPC here — successful dial + immediate close is
// sufficient evidence that a listener exists. The actual Health RPC
// happens later when the caller's Connect runs.
func probeDaemon(ctx context.Context, socketPath string) (dialResult, error) {
	// Pre-flight stat: if the path exists but isn't a socket, classify as
	// hostile before attempting the dial. Linux returns ECONNREFUSED for
	// dialing a regular file (which our errors.Is below would otherwise
	// fold into dialAbsent and trigger an auto-spawn that clobbers the
	// existing file); macOS returns ENOTSOCK. Stat-first removes the OS
	// difference. Mirrors the same check ipc.Listen performs server-side.
	if info, err := os.Stat(socketPath); err == nil && info.Mode()&os.ModeSocket == 0 {
		return dialHostile, fmt.Errorf("probe %s: path exists but is not a socket", socketPath)
	}

	cctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	var d net.Dialer
	conn, err := d.DialContext(cctx, "unix", socketPath)
	if err == nil {
		conn.Close()
		return dialReachable, nil
	}
	// Absent: socket is missing or no listener is bound.
	if errors.Is(err, syscall.ENOENT) || errors.Is(err, syscall.ECONNREFUSED) {
		return dialAbsent, nil
	}
	// Hostile: anything else (EACCES, etc.) — don't spawn.
	return dialHostile, fmt.Errorf("probe %s: %w", socketPath, err)
}

// launchDaemon execs runed as a detached child process, redirecting its
// stdout/stderr to ~/.runed/logs/daemon.log so the parent can exit
// without orphaning a controlling terminal.
func launchDaemon(cfg *Config, socketPath string) error {
	home := filepath.Dir(socketPath)
	if err := os.MkdirAll(filepath.Join(home, "logs"), 0o700); err != nil {
		return fmt.Errorf("mkdir logs: %w", err)
	}
	logPath := filepath.Join(home, "logs", "daemon.log")
	logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open daemon log: %w", err)
	}
	cmd := exec.Command(cfg.RunedBinary)
	env := append(os.Environ(),
		"RUNED_LLAMA_SERVER="+cfg.LlamaServer,
		"RUNED_MODEL="+cfg.Model,
		"RUNED_HOME="+home,
	)
	if cfg.IdleTimeout != "" {
		env = append(env, "RUNED_IDLE_TIMEOUT="+cfg.IdleTimeout)
	}
	cmd.Env = env
	cmd.Stdin = nil
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return fmt.Errorf("exec %s: %w", cfg.RunedBinary, err)
	}
	// Don't Wait — daemon outlives us. logFile stays referenced via the
	// child's fd table, so closing it here doesn't affect the daemon.
	logFile.Close()
	return nil
}

// waitForDaemon polls probeDaemon every 200ms up to timeout.
func waitForDaemon(ctx context.Context, socketPath string, timeout time.Duration) error {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		if r, _ := probeDaemon(cctx, socketPath); r == dialReachable {
			return nil
		}
		select {
		case <-cctx.Done():
			home := filepath.Dir(socketPath)
			return fmt.Errorf("daemon not ready within %s; check %s/logs/daemon.log",
				timeout, home)
		case <-ticker.C:
		}
	}
}
