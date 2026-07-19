// Package runed exposes the repository's license texts as an embedded
// filesystem so the self-bootstrap can install them next to the third-party
// artifacts it downloads (OPS-90). Redistribution of llama-server (MIT) and
// the Qwen3 GGUF weights (Apache-2.0) requires shipping the license texts
// with the copies — embedding them in the daemon binary guarantees every
// machine that receives the artifacts also receives the licenses.
package runed

import "embed"

// LicenseFS holds this repository's own LICENSE plus the third-party license
// texts (with their attribution index). Paths inside the FS mirror the repo:
//
//	LICENSE
//	THIRD_PARTY_LICENSES/README.md
//	THIRD_PARTY_LICENSES/llama.cpp.LICENSE
//	THIRD_PARTY_LICENSES/Qwen3-Embedding.Apache-2.0.LICENSE
//
//go:embed LICENSE THIRD_PARTY_LICENSES
var LicenseFS embed.FS
