#!/usr/bin/env bash
# Standalone self-bootstrap driver. Serves scripts/manifest.dev.json over
# a local HTTP server and builds runed with that URL baked via -ldflags.
# All artifact metadata (URLs, sha256, sizes) lives in manifest.dev.json —
# this script knows where the manifest is, not what's in it.
#
# Usage:
#   bash scripts/dev_standalone.sh            # set up + build
#   bash scripts/dev_standalone.sh --clean    # wipe installed artifacts first
#
# Environment overrides:
#   WORK=/tmp/runed-standalone   (where the HTTP server's docroot lives)
#   PORT=8000                    (HTTP server bind port)
#
# Tear down:
#   kill <pid printed below>     # HTTP server
#   rm -rf ~/.runed/bin/llama-cpp ~/.runed/models ~/.runed/cache/*

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_SRC="$ROOT/scripts/manifest.dev.json"
WORK="${WORK:-/tmp/runed-standalone}"
PORT="${PORT:-8000}"
PLATFORM="$(go env GOOS)-$(go env GOARCH)"

if [[ "$PLATFORM" != "darwin-arm64" ]]; then
    echo "manifest.dev.json only carries darwin-arm64 artifacts; current platform is $PLATFORM." >&2
    echo "Add your platform to manifest.dev.json or run on macOS arm64." >&2
    exit 1
fi

if [[ ! -f "$MANIFEST_SRC" ]]; then
    echo "manifest source missing: $MANIFEST_SRC" >&2
    exit 1
fi

if [[ "${1:-}" == "--clean" ]]; then
    echo "[0/4] cleaning ~/.runed/{bin/llama-cpp,models,cache}"
    rm -rf "$HOME/.runed/bin/llama-cpp" "$HOME/.runed/models" "$HOME/.runed/cache"/*
fi

if grep -q '"sha256": "0000000000000000000000000000000000000000000000000000000000000000"' "$MANIFEST_SRC"; then
    cat <<EOF >&2
manifest.dev.json still contains placeholder values (all-zero sha256s,
example.com URLs). Edit $MANIFEST_SRC with real artifact URLs, SHA-256s,
and sizes before runed can verify the downloads.

Minimum fields to fill:
  platforms.darwin-arm64.llama_server.url / sha256 / size
  models.<variant>.url / sha256 / size

The runed binary itself is fetched separately by 'rune install' and is
not part of this manifest.
EOF
    exit 1
fi

mkdir -p "$WORK"
cp "$MANIFEST_SRC" "$WORK/manifest.json"
echo "[1/3] manifest copied → $WORK/manifest.json (source: $MANIFEST_SRC)"

LOG="$WORK/http-server.log"
( cd "$WORK" && exec python3 -m http.server "$PORT" ) >"$LOG" 2>&1 &
SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null || true
echo "[2/3] http server on http://127.0.0.1:$PORT (pid $SERVER_PID, log $LOG)"

sleep 0.3

cd "$ROOT"
MANIFEST_URL="http://127.0.0.1:$PORT/manifest.json"
echo "[3/4] building runed with DEFAULT_MANIFEST_URL=$MANIFEST_URL"
make build DEFAULT_MANIFEST_URL="$MANIFEST_URL" >/dev/null

# Install symlink so spawn's DefaultRunedBinary (~/.runed/bin/runed) points
# at this fork's freshly-built daemon. Without this, `rundemo` / `client`
# spawning a fresh daemon would either fail (no PATH match) or land on
# whatever stale binary is at that path.
mkdir -p "$HOME/.runed/bin"
ln -sf "$ROOT/bin/runed" "$HOME/.runed/bin/runed"
echo "[4/4] symlinked $HOME/.runed/bin/runed -> $ROOT/bin/runed"

cat <<EOF

ready.

next:
  ./bin/runed
      foreground launch
      first time: downloads tarball (~22 MB) + GGUF (~472 MB) into ~/.runed/
      subsequent: SHA check skips the downloads

  ./bin/rundemo "hello world"
      auto-spawns the daemon via ~/.runed/bin/runed (the symlink), embeds,
      prints the result. Verifies spawn + self-bootstrap end-to-end.

stop the http server when done:
  kill $SERVER_PID

re-trigger the full install path:
  bash scripts/dev_standalone.sh --clean
EOF
