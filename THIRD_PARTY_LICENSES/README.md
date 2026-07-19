# Third-party licenses

`runed` downloads and runs the following third-party components at first
launch (self-bootstrap; see the manifest documentation in the top-level
README). Their license texts are reproduced verbatim in this directory and
are also installed to `$RUNED_HOME/licenses/` alongside the artifacts they
cover, so every machine that receives the binaries receives the licenses.

| Component | Role | License | Upstream |
|---|---|---|---|
| llama.cpp (`llama-server`) | Local inference server child process | MIT — [`llama.cpp.LICENSE`](llama.cpp.LICENSE) | <https://github.com/ggml-org/llama.cpp> |
| Qwen3-Embedding-0.6B (GGUF) | Embedding model weights | Apache-2.0 — [`Qwen3-Embedding.Apache-2.0.LICENSE`](Qwen3-Embedding.Apache-2.0.LICENSE) | <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B> |

Notes:

- The llama.cpp license text is the upstream `LICENSE` file, unmodified
  (`Copyright (c) 2023-2026 The ggml authors`).
- The Qwen3-Embedding-0.6B repositories on Hugging Face declare
  `license: apache-2.0` and ship no LICENSE or NOTICE file of their own, so
  the canonical Apache License 2.0 text is reproduced here unmodified. No
  NOTICE contents exist upstream to reproduce (Apache-2.0 §4(d) applies only
  when the work includes a NOTICE file).
- `runed` itself is licensed under the MIT License — see the top-level
  [`LICENSE`](../LICENSE).
