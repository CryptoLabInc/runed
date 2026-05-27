# Initial cache file consumed via `cmake -C` when building the vendored
# llama.cpp inside this repository. Pinned alongside .llama-cpp-version as
# the single source of truth for how runed builds llama-server.
#
# Goal: a single self-contained llama-server binary with no system
# dependencies beyond the platform's libc/libc++. CPU-only. No web UI.

# Build profile
set(CMAKE_BUILD_TYPE        Release CACHE STRING "")
set(BUILD_SHARED_LIBS       OFF     CACHE BOOL   "")

# Disable optional features that pull in system libraries (curl→openssl,
# openssl directly, embedded web UI via npm/SvelteKit).
set(LLAMA_CURL              OFF     CACHE BOOL   "")
set(LLAMA_OPENSSL           OFF     CACHE BOOL   "")
set(LLAMA_BUILD_UI          OFF     CACHE BOOL   "")

# Build only what we ship. App is the unified `llama` binary; we only need
# `llama-server`. Tests/examples add build time for no gain.
set(LLAMA_BUILD_SERVER      ON      CACHE BOOL   "")
set(LLAMA_BUILD_APP         OFF     CACHE BOOL   "")
set(LLAMA_BUILD_TESTS       OFF     CACHE BOOL   "")
set(LLAMA_BUILD_EXAMPLES    OFF     CACHE BOOL   "")

# Backends: CPU-only. GGML_BLAS defaults OFF here so Linux/Windows stay as
# single self-contained static binaries (avoiding OpenBLAS install on the
# user machine). macOS overrides this to ON in the build invocation, since
# Accelerate.framework ships with the OS — no extra user dependency.
set(GGML_METAL              OFF     CACHE BOOL   "")
set(GGML_CUDA               OFF     CACHE BOOL   "")
set(GGML_ACCELERATE         OFF     CACHE BOOL   "")
set(GGML_BLAS               OFF     CACHE BOOL   "")

# Portability: do not tune to the build machine's CPU. Required for any
# binary that may run on a different CPU than it was built on (CI runners
# generally have newer CPUs than user laptops).
set(GGML_NATIVE             OFF     CACHE BOOL   "")
