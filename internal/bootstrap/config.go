package bootstrap

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

const ConfigVersion = 1

// Config is the parsed shape of $RUNED_HOME/config.json. The file is
// optional; absence is equivalent to a zero-value Config with the current
// Version. Today only ModelVariant is honored; future fields (CtxSize,
// IdleTimeout, custom manifest URL, ...) can fold in without bumping the
// version as long as they remain optional.
//
// Note: the same config.json is also read by internal/spawn (client side)
// with its own schema (LlamaServer, Model, RunedBinary, ...). We do NOT
// DisallowUnknownFields so the two readers can coexist on one file.
type Config struct {
	Version      int    `json:"version"`
	ModelVariant string `json:"model_variant,omitempty"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &Config{Version: ConfigVersion}, nil
		}
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var c Config
	if err := json.NewDecoder(bytes.NewReader(data)).Decode(&c); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if c.Version != ConfigVersion {
		return nil, fmt.Errorf("config %s: version %d not supported (need %d)",
			path, c.Version, ConfigVersion)
	}
	return &c, nil
}
