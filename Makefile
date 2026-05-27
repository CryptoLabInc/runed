.PHONY: all proto build test clean release-tarball llama-server

VERSION ?= v0.1.0-alpha
GOOS ?= $(shell go env GOOS)
GOARCH ?= $(shell go env GOARCH)

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

all: build

proto:
	buf generate

build: proto
	mkdir -p bin
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build \
		-ldflags "-X main.daemonVersion=$(VERSION)" \
		-o bin/runed ./cmd/runed
	GOOS=$(GOOS) GOARCH=$(GOARCH) go build -o bin/rundemo ./cmd/rundemo

# Clone (shallow) and CPU-build llama-server at the pinned ref.
# Unix-only target — Windows CI invokes cmake directly because make/sh aren't
# the natural toolchain there. Reentrant: skips git clone if the directory
# already exists at the right ref, and skips cmake if the binary is fresh.
llama-server:
	@if [ ! -d "$(LLAMA_CPP_DIR)/.git" ]; then \
		mkdir -p $(dir $(LLAMA_CPP_DIR)); \
		git clone --depth 1 --branch $(LLAMA_CPP_REF) \
			https://github.com/ggml-org/llama.cpp $(LLAMA_CPP_DIR); \
	fi
	cmake -B $(LLAMA_CPP_DIR)/build -S $(LLAMA_CPP_DIR) -C $(LLAMA_CPP_CACHE) $(LLAMA_CMAKE_EXTRA)
	# -j2 caps parallel compile jobs: ubuntu-latest (16GB / 4 vCPU) OOM-killed
	# the build at -j auto. ubuntu-24.04-arm and macos-14 survived but the
	# bound is uniform across matrices for predictability.
	cmake --build $(LLAMA_CPP_DIR)/build --target llama-server -j 2 --config Release
	mkdir -p bin
	cp $(LLAMA_SERVER_BIN) bin/llama-server

test:
	go test ./...

clean:
	rm -rf bin/ gen/ dist/

# Packages the Go binaries plus llama-server into a single tarball.
# Assumes `make build` and `make llama-server` (or the workflow's Windows
# equivalent) have already populated bin/.
release-tarball:
	mkdir -p dist
	TARNAME=runed-$(VERSION)-$(GOOS)-$(GOARCH).tar.gz; \
	tar -czf dist/$$TARNAME -C bin runed rundemo llama-server; \
	cd dist && ( \
		(command -v shasum >/dev/null 2>&1 && shasum -a 256 $$TARNAME > $$TARNAME.sha256) \
		|| (command -v sha256sum >/dev/null 2>&1 && sha256sum $$TARNAME > $$TARNAME.sha256) \
		|| { echo "no sha256 tool available" >&2; exit 1; } \
	); \
	echo "Created: dist/$$TARNAME"
