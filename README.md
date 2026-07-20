<p align="center">
  <a href="https://rune.team" aria-label="RUNE website">
    <img src=".github/assets/repo-hero.svg" alt="runed — shared local embedding runtime" width="100%">
  </a>
</p>

<p align="center">
  <img src=".github/assets/repo-badges.svg" alt="Embedding daemon · Qwen3 0.6B · Apache 2.0" width="590">
</p>

<p align="center">
  <a href="https://rune.team">rune.team</a> ·
  <a href="https://rune.team/docs">Documentation</a> ·
  <a href="https://github.com/CryptoLabInc/runed/releases">Releases</a>
</p>

`runed` is RUNE's shared local embedding daemon. It keeps one Qwen3 embedding model per machine and serves every agent session over a Unix domain socket, avoiding a separate model copy for every `rune-mcp` process.

The daemon starts on demand, verifies every downloaded artifact by SHA-256, suspends the heavyweight `llama-server` child when idle, and brings it back transparently on the next embedding request.

## Why a shared daemon

```text
agent session A ── rune-mcp A ─┐
agent session B ── rune-mcp B ─┼── Unix socket ── runed ── llama-server
agent session C ── rune-mcp C ─┘                    └─ one Qwen3 model
```

- **One model per machine** instead of one model per agent session.
- **Local text processing** so passages do not need to leave the user's device for embedding.
- **Automatic startup** serialized across concurrent clients.
- **Idle suspension** that releases model memory without removing the daemon socket.
- **Verified bootstrap** for the `llama-server` archive and GGUF model.
- **Single and batch RPCs** with L2-normalized vectors.

## Installation

End users normally receive `runed` through the [RUNE](https://github.com/CryptoLabInc/rune) installer. The bootstrap layer places the binary under `~/.runed/bin/`, then `rune-mcp` starts it automatically when an embedding is first needed.

For standalone development:

```bash
git clone https://github.com/CryptoLabInc/runed.git
cd runed
make build
./scripts/dev_standalone.sh
```

Linux and macOS are supported. The current client deliberately rejects Windows because the Plan A transport uses a Unix domain socket.

## Go client

Programs can use the public [`client`](client/) package directly:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/CryptoLabInc/runed/client"
)

func main() {
	ctx := context.Background()

	c, err := client.Connect(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	vector, err := c.Embed(ctx, "Why did we cap retry backoff at five minutes?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(vector))
}
```

`client.Connect` first probes the default socket. If no healthy daemon is present, it serializes startup, launches `runed`, waits for health, and reconnects. `client.WithNoSpawn()` is available for tests and explicitly managed deployments.

## Runtime lifecycle

On first boot, `runed` performs the following sequence:

1. Resolve `RUNED_HOME` and the daemon socket.
2. Fetch the release manifest over HTTPS.
3. Download and verify the platform-specific `llama-server` archive.
4. Download and verify the selected Qwen3 GGUF model.
5. Launch `llama-server` on an ephemeral loopback port.
6. Serve gRPC over `~/.runed/embedding.sock`.

Subsequent boots reuse artifacts whose SHA-256 hashes still match the manifest.

After `RUNED_IDLE_TIMEOUT` without an embedding RPC, only the `llama-server` child stops. `runed` stays reachable and restarts the backend for the next `Embed` or `EmbedBatch` call. Set the timeout to `0` to disable idle suspension.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `RUNED_HOME` | Runtime, model, log, and socket directory. | `~/.runed` |
| `RUNED_MANIFEST` | Self-bootstrap manifest URL. | Build-time release URL |
| `RUNED_LLAMA_SERVER` | Use an explicit `llama-server` binary and skip that download. | Unset |
| `RUNED_MODEL` | Use an explicit GGUF model and skip that download. | Unset |
| `RUNED_MODEL_VARIANT` | Select a named model from the manifest. | `default_model` |
| `RUNED_CTX_SIZE` | Maximum input context passed to `llama-server`. | `2048` |
| `RUNED_IDLE_TIMEOUT` | Suspend the model backend after this idle duration. | `10m` |

`RUNED_MANIFEST` should use HTTPS in production. Artifact hashes protect downloads only after the manifest itself has been trusted.

An optional `~/.runed/config.json` can specify `runed_binary`, `llama_server`, `model`, `model_variant`, and `idle_timeout`. Environment variables remain useful for one-off development overrides.

## Release model

The release workflow currently pins **Qwen3-Embedding-0.6B Q8_0**, approximately 610 MB, as the production manifest's default model. The f16 model remains the parity reference; smaller quantizations trade retrieval fidelity for disk and memory savings.

| Variant | Approximate size | Mean cosine vs. f16 | Intended use |
| --- | ---: | ---: | --- |
| f16 | 1.1 GB | ≈ 0.99999 | Parity reference |
| **Q8_0** | **610 MB** | **≈ 0.9993** | **Current release default** |
| Q6_K | 472 MB | 0.994 | Smaller high-quality alternative |
| Q5_K_M | 424 MB | 0.990 | Size-sensitive environments |
| Q4_K_M | 378 MB | 0.971 | Development or secondary reranking |

The selected manifest controls the actual filename, hash, and default. Treat this table as guidance, not a substitute for inspecting a custom manifest.

## Development

Go 1.26.2 or newer is required.

```bash
make build
make test

# Full integration coverage with local artifacts
RUNED_TEST_LLAMA_SERVER=/path/to/llama-server \
RUNED_TEST_GGUF=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf \
go test -race ./...
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for protobuf generation, build-time manifest injection, and model parity requirements.

## Repository map

| Path | Responsibility |
| --- | --- |
| [`cmd/runed/`](cmd/runed/) | Daemon entry point and process lifecycle. |
| [`client/`](client/) | Auto-spawning Go client. |
| [`internal/bootstrap/`](internal/bootstrap/) | Manifest, download, verification, extraction, and license installation. |
| [`internal/backend/`](internal/backend/) | `llama-server` child process and embedding HTTP bridge. |
| [`internal/ipc/`](internal/ipc/) | Local socket listener and path handling. |
| [`internal/server/`](internal/server/) | gRPC health, embedding, centroid, and shutdown services. |
| [`proto/runed/v1/`](proto/runed/v1/) | Public protobuf contract. |

## License and third-party software

- `runed`: [Apache License 2.0](LICENSE) with [NOTICE](NOTICE).
- `llama.cpp` / `llama-server`: MIT; see [`THIRD_PARTY_LICENSES/llama.cpp.LICENSE`](THIRD_PARTY_LICENSES/llama.cpp.LICENSE).
- Qwen3-Embedding-0.6B: Apache License 2.0; see [`THIRD_PARTY_LICENSES/Qwen3-Embedding.Apache-2.0.LICENSE`](THIRD_PARTY_LICENSES/Qwen3-Embedding.Apache-2.0.LICENSE).

The bootstrap installs the applicable texts under `$RUNED_HOME/licenses/` beside the downloaded third-party artifacts. See [`THIRD_PARTY_LICENSES/README.md`](THIRD_PARTY_LICENSES/README.md) for the attribution index.

<p align="center">
  Part of <a href="https://rune.team">RUNE</a> · Built by <a href="https://www.cryptolab.co.kr/">CryptoLab</a>
</p>
