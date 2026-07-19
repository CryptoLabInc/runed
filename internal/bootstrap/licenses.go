package bootstrap

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	runed "github.com/CryptoLabInc/runed"
)

// installLicenses copies the embedded license texts (runed's own LICENSE plus
// the third-party texts for llama-server and the Qwen3 GGUF model) into
// $RUNED_HOME/licenses/, so the machine that just received the downloaded
// artifacts also holds the licenses that cover them (OPS-90). Idempotent —
// files are rewritten in place on every bootstrap, so license updates ship
// with daemon updates.
func installLicenses(p *Paths) error {
	root := filepath.Join(p.Home, "licenses")
	return fs.WalkDir(runed.LicenseFS, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		dst := filepath.Join(root, path)
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		data, err := runed.LicenseFS.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read embedded %s: %w", path, err)
		}
		if err := os.WriteFile(dst, data, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", dst, err)
		}
		return nil
	})
}
