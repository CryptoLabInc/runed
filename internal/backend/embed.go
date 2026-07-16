package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const maxErrorBodyBytes = 4 << 10 // 4 KiB cap for reading an error body

// embedRequestTimeout is an absolute ceiling on a single embed HTTP call.
// http.DefaultClient has no timeout and the caller's ctx may carry none, so
// without this a llama-server that accepts the TCP connection but never
// responds (a hang, not a crash — a crash resets the connection) would block
// the call forever. That call holds inflightMu.RLock, so a forever-blocked
// embed would also wedge restartIfDead's inflightMu.Lock() — the recovery path
// that exists precisely to replace a stuck server. The ceiling is far above
// any legitimate embed, including deep-queue waits observed under heavy
// concurrency (~10s), so it only ever trips on a genuinely wedged backend;
// the request then fails, releasing the RLock, and the next EnsureStarted
// restarts the child. A var (not const) so tests can shrink it.
var embedRequestTimeout = 120 * time.Second

// ErrNotStarted is returned when Embed/EmbedBatch is invoked but the
// backend has no live llama-server (port == 0). Callers can recover by
// invoking EnsureStarted and retrying: the typical trigger is the idle
// ticker stopping the child between an earlier EnsureStarted check and
// the Embed call that follows.
var ErrNotStarted = errors.New("backend not started")

// embedReq is the request shape for llama-server's /v1/embeddings endpoint.
// The endpoint is OpenAI-compatible.
//
// Input is typed as interface{} so callers can pass either a single string
// (→ JSON string) or a []string (→ JSON array). Both shapes are accepted by
// llama-server's OpenAI-compatible /v1/embeddings.
type embedReq struct {
	Input interface{} `json:"input"`
}

// embedResp is the response shape for llama-server's /v1/embeddings endpoint.
type embedResp struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// doJSON POSTs a JSON body and decodes the response into out. It enforces
// two invariants worth flagging to future readers:
//  1. Error responses are truncated at maxErrorBodyBytes so a malicious/
//     buggy upstream cannot exhaust memory.
//  2. On non-happy paths the body is drained to EOF before Close() so HTTP
//     keep-alive connection reuse is preserved.
func (b *LlamaBackend) doJSON(ctx context.Context, path string, in any, out any) error {
	port := b.Port()
	if port == 0 {
		return ErrNotStarted
	}
	url := fmt.Sprintf("http://%s:%d%s", b.cfg.Host, port, path)

	body, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	// Bound the call so a hung (not crashed) llama-server can't block it — and
	// thus the inflightMu.RLock it holds — forever. WithTimeout fires at the
	// earlier of the caller's deadline or our ceiling, so a caller with a
	// tighter deadline still wins.
	ctx, cancel := context.WithTimeout(ctx, embedRequestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("http: %w", err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		slurp, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return fmt.Errorf("llama-server status %d: %s", resp.StatusCode, slurp)
	}

	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	// A completed request is the strongest proof of life there is — it keeps
	// EnsureStarted's health verdict from mistaking a saturated child (whose
	// /health probes starve under inference load) for a dead one.
	b.noteAlive()
	return nil
}

// Embed sends a single text to llama-server's /v1/embeddings endpoint and
// returns the L2-normalized embedding vector.
//
// The normalize parameter is currently ignored because llama-server's
// /v1/embeddings always returns L2-normalized output for embedding models
// with last-pooled output. The parameter is retained in the signature for
// future flexibility (e.g., if llama.cpp exposes an un-normalize flag later).
func (b *LlamaBackend) Embed(ctx context.Context, text string, normalize bool) ([]float32, error) {
	_ = normalize // currently ignored; see godoc above
	b.inflightMu.RLock()
	defer b.inflightMu.RUnlock()
	var out embedResp
	if err := b.doJSON(ctx, "/v1/embeddings", embedReq{Input: text}, &out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 {
		return nil, fmt.Errorf("empty response")
	}
	return out.Data[0].Embedding, nil
}

// EmbedBatch sends multiple texts to /v1/embeddings. llama-server's OpenAI-
// compatible endpoint accepts a JSON array as `input`. Results preserve
// request order.
//
// Empty input returns (nil, nil) — no request is sent. The normalize
// parameter is currently ignored for the same reasons as Embed.
func (b *LlamaBackend) EmbedBatch(ctx context.Context, texts []string, normalize bool) ([][]float32, error) {
	_ = normalize // currently ignored; see Embed godoc
	if len(texts) == 0 {
		return nil, nil
	}
	b.inflightMu.RLock()
	defer b.inflightMu.RUnlock()
	var out embedResp
	if err := b.doJSON(ctx, "/v1/embeddings", embedReq{Input: texts}, &out); err != nil {
		return nil, err
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("batch size mismatch: want %d, got %d", len(texts), len(out.Data))
	}
	result := make([][]float32, len(out.Data))
	for i, d := range out.Data {
		result[i] = d.Embedding
	}
	return result, nil
}
