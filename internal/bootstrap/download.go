package bootstrap

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

var ErrChecksumMismatch = errors.New("checksum mismatch")

// downloadTimeout caps any single artifact transfer. The model alone is
// ~470MB at q6_K; on a healthy network this finishes in seconds-to-minutes,
// so 2 hours is a generous "definitely stalled" threshold rather than a
// realistic SLA. The intent is to guarantee that a wedged HTTP read can't
// keep the daemon process alive indefinitely as a zombie.
const downloadTimeout = 2 * time.Hour

// ProgressFunc is invoked as bytes flow through the download. total is
// -1 when the server doesn't advertise Content-Length. Implementations
// must be cheap (called per ~64KB chunk).
type ProgressFunc func(downloaded, total int64)

// DownloadAndVerify streams url → destPath.partial, computes SHA-256 as
// it goes, then renames into place on match. The partial file is removed
// on any error. expectedSize is sanity-checked when > 0.
func DownloadAndVerify(ctx context.Context, url, expectedSHA256 string, expectedSize int64, destPath string, progress ProgressFunc) error {
	if url == "" || expectedSHA256 == "" {
		return errors.New("download: missing url or sha256")
	}
	dctx, cancel := context.WithTimeout(ctx, downloadTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(dctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("download: build request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download: GET %s: HTTP %d", url, resp.StatusCode)
	}
	// Fail fast on Content-Length mismatch — saves ~470MB of wasted I/O when
	// the manifest and server disagree. ContentLength == -1 means the server
	// didn't advertise a length (chunked transfer); the post-stream check at
	// the bottom of this function is the safety net for that case.
	if expectedSize > 0 && resp.ContentLength > 0 && resp.ContentLength != expectedSize {
		return fmt.Errorf("download: Content-Length mismatch: server claims %d bytes, manifest claims %d",
			resp.ContentLength, expectedSize)
	}

	partial := destPath + ".partial"
	f, err := os.OpenFile(partial, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("download: open partial %s: %w", partial, err)
	}
	committed := false
	defer func() {
		_ = f.Close()
		if !committed {
			_ = os.Remove(partial)
		}
	}()

	h := sha256.New()
	written, err := streamWithProgress(resp.Body, f, h, resp.ContentLength, progress)
	if err != nil {
		return fmt.Errorf("download: write %s: %w", partial, err)
	}
	if expectedSize > 0 && written != expectedSize {
		return fmt.Errorf("download: size mismatch: got %d bytes, manifest claims %d",
			written, expectedSize)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, expectedSHA256) {
		return fmt.Errorf("%w: got %s, want %s", ErrChecksumMismatch, got, expectedSHA256)
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("download: fsync %s: %w", partial, err)
	}
	if err := os.Rename(partial, destPath); err != nil {
		return fmt.Errorf("download: rename %s to %s: %w", partial, destPath, err)
	}
	committed = true
	return nil
}

func streamWithProgress(src io.Reader, dst io.Writer, h io.Writer, total int64, progress ProgressFunc) (int64, error) {
	buf := make([]byte, 64*1024)
	var written int64
	for {
		n, rerr := src.Read(buf)
		if n > 0 {
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return written, werr
			}
			_, _ = h.Write(buf[:n])
			written += int64(n)
			if progress != nil {
				progress(written, total)
			}
		}
		if rerr == io.EOF {
			return written, nil
		}
		if rerr != nil {
			return written, rerr
		}
	}
}

func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("sha256 open %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("sha256 read %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// FileMatchesSHA256 returns (false, nil) for a missing file so the caller
// can treat absence and mismatch identically as "need to (re)download."
func FileMatchesSHA256(path, expected string) (bool, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	got, err := FileSHA256(path)
	if err != nil {
		return false, err
	}
	return strings.EqualFold(got, expected), nil
}
