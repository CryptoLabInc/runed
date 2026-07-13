package server

import (
	"context"
	"testing"

	runedv1 "github.com/CryptoLabInc/runed/gen/runed/v1"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// reasonOf extracts the ErrorInfo reason from a status error ("" if none).
func reasonOf(err error) string {
	st, ok := status.FromError(err)
	if !ok {
		return ""
	}
	for _, d := range st.Details() {
		if info, ok := d.(*errdetails.ErrorInfo); ok {
			return info.GetReason()
		}
	}
	return ""
}

// The two precondition conditions must be machine-distinguishable by reason
// while keeping the FAILED_PRECONDITION code (transport-level non-retry stays;
// see TestServer_EmbedFailsBeforeBackendSet's rationale).
func TestPreconditionErrCarriesReason(t *testing.T) {
	for _, tc := range []struct{ reason, msg string }{
		{ReasonBootstrapping, "daemon is bootstrapping; embed not yet available"},
		{ReasonNoCentroidSet, "no centroid set loaded"},
	} {
		err := preconditionErr(tc.reason, tc.msg)
		if got := status.Code(err); got != codes.FailedPrecondition {
			t.Fatalf("%s: code = %v, want FailedPrecondition", tc.reason, got)
		}
		if got := reasonOf(err); got != tc.reason {
			t.Fatalf("reason = %q, want %q", got, tc.reason)
		}
	}
}

// The bootstrap window (backend not yet wired) must surface the BOOTSTRAPPING
// reason over the wire — this is what lets rune-mcp wait-and-retry instead of
// misfiring the centroid resync.
func TestEmbedBootstrapReasonOnWire(t *testing.T) {
	s := New("test")
	_, err := s.Embed(context.Background(), &runedv1.EmbedRequest{Text: "x"})
	if got := reasonOf(err); got != ReasonBootstrapping {
		t.Fatalf("Embed reason = %q, want %q", got, ReasonBootstrapping)
	}
	_, err = s.EmbedBatch(context.Background(), &runedv1.EmbedBatchRequest{Texts: []string{"x"}})
	if got := reasonOf(err); got != ReasonBootstrapping {
		t.Fatalf("EmbedBatch reason = %q, want %q", got, ReasonBootstrapping)
	}
}
