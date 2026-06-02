.PHONY: all proto build test clean release-tarball llama-server

VERSION ?= v0.1.0-alpha
GOOS ?= $(shell go env GOOS)
GOARCH ?= $(shell go env GOARCH)

# OS_LABEL names the build OS in the release tarball (e.g. ubuntu-2204, mac-14)
# so the artifact records its glibc/SDK baseline. CI passes the running image's
# real identity; local `make release-tarball` falls back to GOOS.
OS_LABEL ?= $(GOOS)

LLAMA_CPP_REF := $(shell cat .llama-cpp-version)
LLAMA_CPP_DIR := third_party/llama.cpp
LLAMA_CPP_CACHE := $(CURDIR)/ci/llama-cpp-cache.cmake
LLAMA_SERVER_BIN := $(LLAMA_CPP_DIR)/build/bin/llama-server

# macOS-only: turn on BLAS so llama.cpp links Accelerate.framework. It ships
# with the OS, so this adds no user dependency. Linux/Windows stay BLAS=OFF
# (cache default) to keep the static single-binary property without pulling
# in OpenBLAS.
LLAMA_CMAKE_EXTRA :=
ifeq ($(shell uname -s),Darwin)
LLAMA_CMAKE_EXTRA += -DGGML_BLAS=ON
endif

# DEFAULT_MANIFEST_URL is baked into the binary so end users don't need to
# set RUNED_MANIFEST themselves. Leave empty for dev builds (env-required);
# release builds should pass the production URL.
DEFAULT_MANIFEST_URL ?=

# DEFAULT_RUNED_BINARY is the fallback path the spawn package uses when no
# `runed` is on PATH and config.json doesn't specify one. ~/.runed/bin/runed
# matches where `rune install` places it, so dev and release builds share
# the same default — dev builds expect you to symlink your fork binary
# into that location (scripts/dev_standalone.sh does this for you).
DEFAULT_RUNED_BINARY ?= $(HOME)/.runed/bin/runed

LDFLAGS := -X main.daemonVersion=$(VERSION)
ifneq ($(DEFAULT_MANIFEST_URL),)
LDFLAGS += -X github.com/CryptoLabInc/runed/internal/bootstrap.DefaultManifestURL=$(DEFAULT_MANIFEST_URL)
endif
ifneq ($(DEFAULT_RUNED_BINARY),)
LDFLAGS += -X github.com/CryptoLabInc/runed/internal/spawn.DefaultRunedBinary=$(DEFAULT_RUNED_BINARY)
endif

all: build

proto:
	buf generate

build: proto
	mkdir -p bin
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build \
		-ldflags "$(LDFLAGS)" \
		-o bin/runed ./cmd/runed
	# rundemo (and any future client binary) needs the same ldflags so spawn's
	# DefaultRunedBinary / DefaultManifestURL are reachable from the client side.
	# Unknown -X targets (main.daemonVersion isn't defined in rundemo) are silently
	# ignored by the linker.
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build \
		-ldflags "$(LDFLAGS)" \
		-o bin/rundemo ./cmd/rundemo

# Clone (shallow) and CPU-build llama-server at the pinned ref.
# Reentrant: skips git clone if the directory already exists at the right ref,
# and skips cmake if the binary is fresh.
llama-server:
	@if [ ! -d "$(LLAMA_CPP_DIR)/.git" ]; then \
		mkdir -p $(dir $(LLAMA_CPP_DIR)); \
		git clone --depth 1 --branch $(LLAMA_CPP_REF) \
			https://github.com/ggml-org/llama.cpp $(LLAMA_CPP_DIR); \
	fi
	cmake -B $(LLAMA_CPP_DIR)/build -S $(LLAMA_CPP_DIR) -C $(LLAMA_CPP_CACHE) $(LLAMA_CMAKE_EXTRA)
	# -j2 caps parallel compile jobs: ubuntu-22.04 (16GB / 4 vCPU) OOM-killed
	# the build at -j auto. ubuntu-22.04-arm and macos-14 survived but the
	# bound is uniform across matrices for predictability.
	cmake --build $(LLAMA_CPP_DIR)/build --target llama-server -j 2 --config Release
	mkdir -p bin
	cp $(LLAMA_SERVER_BIN) bin/llama-server

test:
	go test ./...

clean:
	rm -rf bin/ gen/ dist/

# Packages the Go binaries plus llama-server into a single tarball.
# Assumes `make build` and `make llama-server` have already populated bin/.
release-tarball:
	mkdir -p dist
	TARNAME=runed-$(VERSION)-$(OS_LABEL)-$(GOARCH).tar.gz; \
	tar -czf dist/$$TARNAME -C bin runed rundemo llama-server; \
	cd dist && ( \
		(command -v shasum >/dev/null 2>&1 && shasum -a 256 $$TARNAME > $$TARNAME.sha256) \
		|| (command -v sha256sum >/dev/null 2>&1 && sha256sum $$TARNAME > $$TARNAME.sha256) \
		|| { echo "no sha256 tool available" >&2; exit 1; } \
	); \
	echo "Created: dist/$$TARNAME"
