// Package client is the Go client library for the runed embedding daemon.
//
// Connect transparently auto-spawns the daemon if none is reachable, using
// ~/.runed/config.json for the spawn paths. Use WithNoSpawn() to disable
// the auto-spawn fallback in test or manual operation.
//
// Typical usage:
//
//	c, err := client.Connect(ctx)
//	if err != nil { ... }
//	defer c.Close()
//	vec, err := c.Embed(ctx, "hello world")
//
// The daemon is expected to be listening at ~/.runed/embedding.sock on
// macOS/Linux. Override via WithSocketPath for tests or non-default installs.
package client

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"github.com/CryptoLabInc/runed/internal/ipc"
	"github.com/CryptoLabInc/runed/internal/spawn"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// Client is a connected runed daemon client. Not safe for concurrent close but
// individual RPC calls are safe for concurrent use (gRPC client pools streams).
type Client struct {
	// mu serializes the retry-on-Unavailable path so that concurrent
	// Embed/EmbedBatch calls against a stale daemon don't all race to
	// re-spawn and reconnect. Held only inside the retry branch; the
	// happy path (non-Unavailable RPCs) is unlocked.
	mu sync.Mutex

	conn *grpc.ClientConn
	grpc runedv1.RunedServiceClient
	// socketPath is the CANONICAL path captured at Connect time for
	// retry/respawn (T9). Dial sites resolve it via ipc.ResolveSocketPath
	// (INST-7); spawn.EnsureDaemon needs the canonical form because it
	// derives the daemon's RUNED_HOME from the path's parent directory.
	socketPath string
	noSpawn    bool // captured to avoid retrying when caller opted out (T9)
}

// Option configures Connect behavior.
type Option func(*options)

type options struct {
	socketPath string
	noSpawn    bool
}

// WithSocketPath overrides the default socket path (~/.runed/embedding.sock).
// Primarily for tests and non-default installations.
func WithSocketPath(p string) Option { return func(o *options) { o.socketPath = p } }

// WithNoSpawn disables the auto-spawn fallback path. When passed,
// Connect surfaces the underlying dial error instead of trying to
// start a daemon. Use for tests, manual operation, or when the caller
// wants explicit control over daemon lifecycle.
func WithNoSpawn() Option {
	return func(o *options) { o.noSpawn = true }
}

// Connect dials the runed daemon at ~/.runed/embedding.sock (UDS) and verifies
// connectivity by issuing a Health RPC. Windows support is deferred to Plan B.
//
// If the initial dial fails and WithNoSpawn was not passed, Connect invokes
// spawn.EnsureDaemon (which reads ~/.runed/config.json, serializes via flock,
// execs the daemon detached, and polls Health) and then retries the dial.
func Connect(ctx context.Context, opts ...Option) (*Client, error) {
	if runtime.GOOS == "windows" {
		return nil, fmt.Errorf("runed client: Windows support not in Plan A")
	}
	o := &options{}
	for _, fn := range opts {
		fn(o)
	}
	if o.socketPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("home: %w", err)
		}
		o.socketPath = filepath.Join(home, ".runed", "embedding.sock")
	}
	c, err := dialAndProbe(ctx, o.socketPath)
	if err == nil {
		c.noSpawn = o.noSpawn
		return c, nil
	}
	if o.noSpawn {
		return nil, err
	}
	if spawnErr := spawn.EnsureDaemon(ctx, o.socketPath); spawnErr != nil {
		return nil, fmt.Errorf("auto-spawn: %w", spawnErr)
	}
	c, err = dialAndProbe(ctx, o.socketPath)
	if err != nil {
		return nil, err
	}
	c.noSpawn = o.noSpawn
	return c, nil
}

// dialAndProbe opens the UDS gRPC connection and verifies the daemon
// responds to Health. Extracted from the original inline Connect body so
// it can be invoked both for the initial dial and the post-spawn retry.
//
// grpc-go natively resolves "unix://" targets to a Unix socket dialer;
// no custom grpc.WithContextDialer is required. Using both causes the
// dialer to receive the full "unix://..." URI as its addr, which then
// becomes "unix:///path" after the net.Dialer prepends no scheme — the
// socket file "unix:///path" of course does not exist.
//
// The canonical socketPath is resolved just-in-time for the dial (INST-7):
// when it exceeds the sun_path limit the daemon binds a short deterministic
// alias, and connect() on the canonical path would fail with EINVAL. The
// Client keeps the canonical form so respawn derives the right RUNED_HOME.
func dialAndProbe(ctx context.Context, socketPath string) (*Client, error) {
	conn, err := grpc.NewClient(
		"unix://"+ipc.ResolveSocketPath(socketPath),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("grpc dial: %w", err)
	}
	c := &Client{
		conn:       conn,
		grpc:       runedv1.NewRunedServiceClient(conn),
		socketPath: socketPath,
	}
	// Eagerly verify connectivity via Health RPC. This is where a missing
	// daemon / stale socket would surface before the caller issues real work.
	if _, err := c.grpc.Health(ctx, &runedv1.HealthRequest{}); err != nil {
		conn.Close()
		return nil, fmt.Errorf("health check: %w", err)
	}
	return c, nil
}

// reconnectLocked closes the current gRPC connection and dials a new one
// to the same socket. Caller MUST hold c.mu. Used only inside the retry
// branch of Embed/EmbedBatch.
func (c *Client) reconnectLocked() error {
	c.conn.Close()
	conn, err := grpc.NewClient(
		"unix://"+ipc.ResolveSocketPath(c.socketPath), // INST-7: dial the alias
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return fmt.Errorf("reconnect dial: %w", err)
	}
	c.conn = conn
	c.grpc = runedv1.NewRunedServiceClient(conn)
	return nil
}

// Embed returns an L2-normalized embedding for a single text.
//
// When the daemon has idle-exited between RPC calls the underlying gRPC
// stub returns codes.Unavailable; rather than surfacing that to the
// caller we re-spawn the daemon, reconnect the gRPC conn, and retry the
// RPC once. Disable via WithNoSpawn at Connect time.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := c.grpc.Embed(ctx, &runedv1.EmbedRequest{Text: text})
	if err != nil && status.Code(err) == codes.Unavailable && !c.noSpawn {
		// Serialize the retry critical section. After acquiring the lock,
		// re-issue the call: another goroutine may have already done the
		// spawn+reconnect dance during our wait, in which case the call
		// will succeed without us touching anything.
		c.mu.Lock()
		defer c.mu.Unlock()
		resp, err = c.grpc.Embed(ctx, &runedv1.EmbedRequest{Text: text})
		if err == nil || status.Code(err) != codes.Unavailable {
			// Either it now works or it's a different error class — no respawn needed.
			if err != nil {
				return nil, err
			}
			return resp.Vector, nil
		}
		// Still Unavailable under the lock. We're the goroutine that
		// does the work.
		if e := spawn.EnsureDaemon(ctx, c.socketPath); e != nil {
			return nil, fmt.Errorf("embed: re-spawn failed: %w", e)
		}
		if e := c.reconnectLocked(); e != nil {
			return nil, fmt.Errorf("embed: reconnect failed: %w", e)
		}
		resp, err = c.grpc.Embed(ctx, &runedv1.EmbedRequest{Text: text})
	}
	if err != nil {
		return nil, err
	}
	return resp.Vector, nil
}

// EmbedBatch returns L2-normalized embeddings for multiple texts in one RPC.
// Order matches the input.
//
// Symmetric retry-on-Unavailable behavior with Embed; see that doc comment.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	resp, err := c.grpc.EmbedBatch(ctx, &runedv1.EmbedBatchRequest{Texts: texts})
	if err != nil && status.Code(err) == codes.Unavailable && !c.noSpawn {
		c.mu.Lock()
		defer c.mu.Unlock()
		resp, err = c.grpc.EmbedBatch(ctx, &runedv1.EmbedBatchRequest{Texts: texts})
		if err == nil || status.Code(err) != codes.Unavailable {
			if err != nil {
				return nil, err
			}
			out := make([][]float32, len(resp.Embeddings))
			for i, e := range resp.Embeddings {
				out[i] = e.Vector
			}
			return out, nil
		}
		if e := spawn.EnsureDaemon(ctx, c.socketPath); e != nil {
			return nil, fmt.Errorf("embed batch: re-spawn failed: %w", e)
		}
		if e := c.reconnectLocked(); e != nil {
			return nil, fmt.Errorf("embed batch: reconnect failed: %w", e)
		}
		resp, err = c.grpc.EmbedBatch(ctx, &runedv1.EmbedBatchRequest{Texts: texts})
	}
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(resp.Embeddings))
	for i, e := range resp.Embeddings {
		out[i] = e.Vector
	}
	return out, nil
}

// Info returns daemon metadata (version, model identity, vector dim, etc).
func (c *Client) Info(ctx context.Context) (*runedv1.InfoResponse, error) {
	return c.grpc.Info(ctx, &runedv1.InfoRequest{})
}

// Close releases resources held by the Client. Subsequent RPC calls will fail.
func (c *Client) Close() error {
	return c.conn.Close()
}
