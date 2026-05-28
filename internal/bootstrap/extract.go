package bootstrap

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ExtractTarGz extracts srcPath into destDir, creating destDir if needed.
// Regular files are written; directories are created with the archived
// mode bits ORed against 0o700; symlinks, hardlinks, devices and any
// entry whose resolved path escapes destDir are silently skipped.
//
// Returns the list of regular files written, in archive order.
func ExtractTarGz(srcPath, destDir string) ([]string, error) {
	src, err := os.Open(srcPath)
	if err != nil {
		return nil, fmt.Errorf("extract: open %s: %w", srcPath, err)
	}
	defer src.Close()
	gz, err := gzip.NewReader(src)
	if err != nil {
		return nil, fmt.Errorf("extract: gzip reader: %w", err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)

	if err := os.MkdirAll(destDir, 0o700); err != nil {
		return nil, fmt.Errorf("extract: mkdir %s: %w", destDir, err)
	}
	destAbs, err := filepath.Abs(destDir)
	if err != nil {
		return nil, fmt.Errorf("extract: abs %s: %w", destDir, err)
	}

	var extracted []string
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return extracted, fmt.Errorf("extract: tar next: %w", err)
		}
		// filepath.IsLocal (Go 1.20+) rejects absolute paths, any segment
		// that would escape via "..", empty paths, and Windows reserved/UNC
		// forms in one shot — replaces the prior strings.Contains + Rel
		// pair, which also over-rejected legitimate names like "foo..bar".
		// Sufficient here because the archive is fetched via a trusted
		// manifest and SHA-256 verified before extraction.
		name := filepath.FromSlash(hdr.Name)
		if !filepath.IsLocal(name) {
			continue
		}
		dest := filepath.Join(destAbs, name)
		switch hdr.Typeflag {
		case tar.TypeDir:
			mode := os.FileMode(hdr.Mode) & 0o777
			if mode == 0 {
				mode = 0o700
			} else {
				mode |= 0o700
			}
			if err := os.MkdirAll(dest, mode); err != nil {
				return extracted, fmt.Errorf("extract: mkdir %s: %w", dest, err)
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(dest), 0o700); err != nil {
				return extracted, fmt.Errorf("extract: mkdir parent %s: %w", dest, err)
			}
			if err := writeOne(tr, dest, hdr.Mode); err != nil {
				return extracted, fmt.Errorf("extract: %s: %w", name, err)
			}
			extracted = append(extracted, dest)
		default:
			// Skip symlinks, hardlinks, devices. llama.cpp release tarballs
			// don't need them; allowing them would broaden the attack surface
			// without a real use case.
			continue
		}
	}
	return extracted, nil
}

func writeOne(r io.Reader, dest string, mode int64) error {
	tmp := dest + ".extract.partial"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("open tmp %s: %w", tmp, err)
	}
	if _, err := io.Copy(f, r); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("copy: %w", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("fsync: %w", err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	effective := os.FileMode(mode) & 0o777
	if effective == 0 {
		effective = 0o600
	}
	if err := os.Chmod(tmp, effective); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("chmod: %w", err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename %s to %s: %w", tmp, dest, err)
	}
	return nil
}
