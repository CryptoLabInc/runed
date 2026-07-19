// Package runed exposes the repository's license texts as an embedded
// filesystem so the self-bootstrap can install them next to the third-party
// artifacts it downloads. Redistribution of llama-server (MIT) and
// the Qwen3 GGUF weights (Apache-2.0) requires shipping the license texts
// with the copies — embedding them in the daemon binary guarantees every
// machine that receives the artifacts also receives the licenses.
package runed

import "embed"

// LicenseFS holds this repository's own LICENSE (Apache-2.0) and NOTICE plus
// the third-party license texts (with their attribution index). Apache-2.0
// §4(d) requires redistributions to carry the NOTICE file, so it travels with
// the LICENSE everywhere. Paths inside the FS mirror the repo:
//
//	LICENSE
//	NOTICE
//	THIRD_PARTY_LICENSES/README.md
//	THIRD_PARTY_LICENSES/llama.cpp.LICENSE
//	THIRD_PARTY_LICENSES/Qwen3-Embedding.Apache-2.0.LICENSE
//
//go:embed LICENSE NOTICE THIRD_PARTY_LICENSES
var LicenseFS embed.FS
