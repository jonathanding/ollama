package ggml

/*
#include "ggml.h"
#include "ggml-backend.h"

extern void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out);
*/
import "C"

import (
	"time"
	"unsafe"

	"github.com/ollama/ollama/llm/profiler"
)

// EnableTiming enables/disables per-node timing collection on all backends.
func (b *Backend) EnableTiming(enabled bool) {
	C.ggml_backend_sched_set_timing(b.sched, C.bool(enabled))
}

// CollectTiming reads per-node timing data from all splits after sync.
// Must be called after synchronize and before next graph_compute.
func (b *Backend) CollectTiming(passStartTime time.Time) []profiler.OpEvent {
	// Get the scheduler's internal graph copy — valid between graph_compute and next graph_compute.
	// This is a struct member on the sched (not a heap pointer), so its lifetime = sched lifetime.
	// No dangling pointer risk, unlike storing a Context graph pointer on the Backend.
	graph := C.ggml_backend_sched_get_graph(b.sched)
	if graph == nil {
		return nil
	}

	nSplits := int(C.ggml_backend_sched_get_n_splits(b.sched))
	var events []profiler.OpEvent

	for s := 0; s < nSplits; s++ {
		nNodes := int(C.ggml_backend_sched_get_split_n_nodes(b.sched, C.int(s)))
		if nNodes == 0 {
			continue
		}
		iStart := int(C.ggml_backend_sched_get_split_start(b.sched, C.int(s)))

		buf := make([]uint64, nNodes)
		n := int(C.ggml_backend_sched_get_split_timing(
			b.sched, C.int(s),
			(*C.uint64_t)(unsafe.Pointer(&buf[0])), C.int(nNodes)))
		if n == 0 {
			continue
		}

		for j := 0; j < n; j++ {
			elapsed := buf[j]
			if elapsed == 0 {
				continue
			}

			// Get the node from the scheduler's graph copy via split offset
			nodeIdx := iStart + j
			if nodeIdx >= int(C.ggml_graph_n_nodes(graph)) {
				continue
			}
			node := C.ggml_graph_node(graph, C.int(nodeIdx))

			var info C.struct_ggml_node_info
			C.ggml_node_get_info(node, &info)

			ev := profiler.OpEvent{
				Type:    "op",
				Op:      C.GoString(info.op_name),
				Name:    C.GoString(info.tensor_name),
				DType:   C.GoString(info.dtype_name),
				Backend: C.GoString(info.backend_name),
				TEnd:    int64(elapsed), // temporary: holds duration, reconstructed below
			}

			// Extract shape
			for d := 0; d < 4; d++ {
				ev.OutShape = append(ev.OutShape, int64(info.shape[d]))
			}

			// Extract source names
			for k := 0; k < int(info.n_srcs); k++ {
				if info.src_names[k] != nil {
					ev.SrcNames = append(ev.SrcNames, C.GoString(info.src_names[k]))
				}
			}

			events = append(events, ev)
		}
	}

	// Reconstruct absolute timestamps from cumulative elapsed
	cursor := passStartTime.UnixNano()
	for i := range events {
		events[i].TStart = cursor
		events[i].TEnd = cursor + events[i].TEnd
		cursor = events[i].TEnd
	}

	return events
}
