# Contributing to runed

## Prerequisites

- `llama-server` binary built from a recent llama.cpp (must emit
  `qwen3.rope.freq_base = 1000000` metadata in converted GGUFs; anything
  from April 2026 onward is safe).
- Qwen3-Embedding-0.6B GGUF, **q6_K** for production (or **f16** for parity
  testing — see the Model variants table in the README).
- Go 1.26+ for building from source (see `go.mod`).
- [`buf`](https://buf.build/docs/installation) CLI for proto codegen.

## Build

```bash
make proto         # generate Go stubs from proto/ into gen/
make build         # produce bin/runed and bin/rundemo
```

### Build-time options

Release / dev builds inject default URLs via `-ldflags` so end users
never need to discover environment variables:

```bash
make build \
    DEFAULT_MANIFEST_URL=https://your.host/runed-manifest.json \
    DEFAULT_RUNED_BINARY=$HOME/.runed/bin/runed
```

- `DEFAULT_MANIFEST_URL` → `internal/bootstrap.DefaultManifestURL`,
  consumed by the daemon's self-bootstrap when `RUNED_MANIFEST` is unset.
- `DEFAULT_RUNED_BINARY` → `internal/spawn.DefaultRunedBinary`,
  consumed by clients (e.g. `rune-mcp`, `rundemo`) when they need to
  auto-spawn the daemon and neither config nor PATH resolves a binary.

## Testing

```bash
make test                                     # unit + compile-time tests

RUNED_TEST_LLAMA_SERVER=/path/to/llama-server \
RUNED_TEST_GGUF=$PWD/models/qwen3-embedding-0.6b.f16.gguf \
go test -race ./...                           # full suite including integration
```

## GGUF conversion gotcha

The model's GGUF must carry `qwen3.rope.freq_base = 1000000` metadata or
llama.cpp will fall back to theta 10000, producing wrong positional encoding
and 3–5% cosine drift that scales with token count. Use a recent llama.cpp
(April 2026 or later) when converting, and re-verify parity if you downgrade.

## Directory layout

```
runed/
├── cmd/
│   ├── runed/          # daemon entrypoint
│   └── rundemo/        # tiny smoke-test consumer
├── client/             # Go client library (external import)
├── gen/                # generated gRPC stubs (checked in)
├── internal/
│   ├── backend/        # llama-server subprocess + HTTP Embed/EmbedBatch
│   ├── ipc/            # UDS listener
│   ├── protosmoke/     # proto compile smoke test
│   └── server/         # gRPC RunedService handlers
├── proto/runed/v1/     # proto definitions
└── Makefile
```
