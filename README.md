# runed — Shared embedding daemon

`runed` is a Go daemon that wraps [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` to serve Qwen3-Embedding-0.6B embeddings via gRPC over a UNIX
domain socket. It is designed as a **shared singleton process per machine** so
that multiple client sessions do not each load their own ~400 MB embedding
model.

## Installation

`runed` is normally installed via the [`rune` CLI](https://github.com/CryptoLabInc/rune):

```
rune install
```

This places the binary at `~/.runed/bin/runed` with a manifest URL baked
in at build time (via `-ldflags`). End users don't need to set any
environment variable — the first launch downloads the `llama-server`
release tarball and the embedding GGUF described by the manifest into
`~/.runed/{bin/llama-cpp,models}/`. Subsequent launches reuse the
installed artifacts as long as the manifest SHA-256s still match.

For standalone testing of the daemon without `rune`, see
[`scripts/dev_standalone.sh`](scripts/dev_standalone.sh).

## Usage

`runed` is a passive gRPC daemon. It is normally spawned on demand by a
client (e.g. `rune-mcp`, `rundemo`, or any program that imports the
`client/` package) the first time an embedding request arrives:

```
client.Connect()
    socket dial fails
    → spawn.EnsureDaemon execs ~/.runed/bin/runed (detached)
       → paths.Resolve / EnsureDirs
       → socket probe (exit 0 if another daemon already listening)
       → self-bootstrap (manifest fetch → llama-server tarball + GGUF
         download on first boot, SHA cache check on subsequent boots)
       → backend.Start (llama-server child, port 0)
       → listen on ~/.runed/embedding.sock
    client reconnects → embed RPC → result
```

Direct foreground launch is supported for development and debugging:

```
./bin/runed
```

After `RUNED_IDLE_TIMEOUT` (default 10m) of no RPC activity, the daemon
stops its `llama-server` child to release the ~470MB+ of model weights
from memory — but `runed` itself stays up with the gRPC socket still
listening. The next `Embed`/`EmbedBatch` RPC transparently restarts the
backend (cold-start latency paid only on that single request). The
daemon process exits only on SIGINT/SIGTERM/SIGHUP or a Shutdown RPC.

## Configuration

### Environment

| Variable | Purpose | Default |
|---|---|---|
| `RUNED_HOME`          | Data directory                                              | `~/.runed` |
| `RUNED_MANIFEST`      | Manifest URL for self-bootstrap                             | `DefaultManifestURL` (build-time ldflags) |
| `RUNED_LLAMA_SERVER`  | Skip self-bootstrap; use this binary                        | (unset) |
| `RUNED_MODEL`         | Skip self-bootstrap; use this GGUF                          | (unset) |
| `RUNED_MODEL_VARIANT` | Pick a non-default model from `manifest.models`             | `manifest.default_model` |
| `RUNED_CTX_SIZE`      | `llama-server --ctx-size` (max input length in tokens)      | 2048 |
| `RUNED_IDLE_TIMEOUT`  | After this much idle, stop the llama-server child to free model memory. `runed` itself stays up; the next Embed RPC resurrects the backend. `"0"` disables suspend | 10m |

`RUNED_MANIFEST` should be **HTTPS** in production. HTTP is permitted
(for private networks) but emits a warning at startup — a MITM that
rewrites the manifest can also rewrite the per-artifact SHA256s, so
artifact integrity collapses to "trust the manifest channel" alone.

The `DefaultManifestURL` referenced in the table above is injected at
build time via `-ldflags`; see [`CONTRIBUTING.md`](CONTRIBUTING.md#build-time-options)
for the relevant `make build` flags.

### Manifest format

`runed` reads only a subset of the manifest; extra keys (used by the
companion `rune` installer) are ignored:

```json
{
  "version": 1,
  "platforms": {
    "darwin-arm64": {
      "llama_server": {
        "url":     "https://.../llama-mac-arm64.tar.gz",
        "sha256":  "...",
        "size":    22000000,
        "extract": "tar.gz",
        "exec":    "llama-server"
      }
    }
  },
  "models": {
    "qwen3-embedding-0.6b.q6_K": {
      "url":    "https://huggingface.co/.../qwen3-embedding-0.6b-q6_k.gguf",
      "sha256": "...",
      "size":   472000000
    }
  },
  "default_model": "qwen3-embedding-0.6b.q6_K"
}
```

- `extract`: `""` (raw binary placed at `LlamaDir/<exec>`) or `"tar.gz"`
  (extracted into `LlamaDir`, `exec` path resolved inside).
- A sidecar marker `~/.runed/bin/llama-cpp/.llama_server.sha256` tracks
  the last-installed tarball hash so repeat boots don't re-extract.
- Models are verified by hashing the on-disk file against
  `models[<variant>].sha256` — no marker file.

### config.json (`~/.runed/config.json`, optional)

This file is read by **two** components with overlapping schemas; leave
any field you don't need unset. Unknown fields are ignored.

| Field | Reader | Purpose |
|---|---|---|
| `version`        | both        | Schema version (currently `1`) |
| `llama_server`   | spawn       | If set, skip daemon self-bootstrap and use this binary |
| `model`          | spawn       | If set, skip daemon self-bootstrap and use this GGUF |
| `runed_binary`   | spawn       | Path to the daemon binary (fallback to PATH / `DefaultRunedBinary`) |
| `idle_timeout`   | spawn       | Propagated to the spawned daemon as `RUNED_IDLE_TIMEOUT` |
| `model_variant`  | bootstrap   | Pick a manifest model variant (overrides `manifest.default_model`) |

Example — minimal config telling the spawn layer to use a custom
`runed_binary` but leaving artifact resolution to self-bootstrap:

```json
{
  "version": 1,
  "runed_binary": "/opt/runed/bin/runed",
  "idle_timeout": "1m"
}
```

## Model variants

The f16 GGUF is the parity reference — cosine similarity ≥ 0.9999 against
sentence-transformers on all 8 fixture texts. Quantized variants trade parity
for size/latency.

| Variant | Size | Mean cosine | Verdict |
|---|---|---|---|
| f16 | 1.1 GB | ≈ 0.99999 | parity reference |
| q8_0 | 610 MB | ≈ 0.9993 | high-fidelity alternative |
| q6_K | 472 MB | 0.994 | **production default** |
| q5_K_M | 424 MB | 0.990 | borderline; natural-language only |
| q4_K_M | 378 MB | 0.971 | rerank-only or dev staging; not sole retrieval backbone |

**Production default is q6_K** (`models/qwen3-embedding-0.6b.q6_K.gguf`) —
23% smaller than q8_0 with mean cosine 0.994 against the f16 reference.
f16 is reserved for parity verification against sentence-transformers.
Set the daemon's `RUNED_MODEL` env var to the desired GGUF path.

## License

- `runed` Go code: MIT (LICENSE file forthcoming).
- Qwen3-Embedding-0.6B: Apache 2.0.
- llama.cpp: MIT.

Redistribution of the bundled `llama-server` binary and GGUF model files is
permitted under the respective upstream licenses; CryptoLab makes no
independent claims on those artifacts.
