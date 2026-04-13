package llama

/*
#include "llama.h"
#include "ggml-backend.h"

extern void ggml_node_get_info(struct ggml_tensor * node, struct ggml_node_info * out);
*/
import "C"

import (
	"time"
	"unsafe"

	"github.com/ollama/ollama/llm/profiler"
)

// EnableTiming enables/disables per-node timing on this llama context.
func (c *Context) EnableTiming(enabled bool) {
	C.llama_context_enable_timing(c.c, C.bool(enabled))
}

// CollectTiming reads per-node timing data after Synchronize.
func (c *Context) CollectTiming(passStartTime time.Time) []profiler.OpEvent {
	nSplits := int(C.llama_context_get_n_splits(c.c))
	var events []profiler.OpEvent

	graph := C.llama_context_get_graph(c.c)
	if graph == nil {
		return nil
	}

	for s := 0; s < nSplits; s++ {
		nNodes := int(C.llama_context_get_split_n_nodes(c.c, C.int(s)))
		if nNodes == 0 {
			continue
		}
		iStart := int(C.llama_context_get_split_start(c.c, C.int(s)))

		buf := make([]uint64, nNodes)
		n := int(C.llama_context_get_split_timing(
			c.c, C.int(s),
			(*C.uint64_t)(unsafe.Pointer(&buf[0])), C.int(nNodes)))
		if n == 0 {
			continue
		}

		for j := 0; j < n; j++ {
			elapsed := buf[j]
			if elapsed == 0 {
				continue
			}

			nodeIdx := iStart + j
			if C.int(nodeIdx) >= graph.n_nodes {
				continue
			}
			nodes := unsafe.Slice(graph.nodes, graph.n_nodes)
			node := nodes[nodeIdx]

			var info C.struct_ggml_node_info
			C.ggml_node_get_info(node, &info)

			ev := profiler.OpEvent{
				Type:    "op",
				Op:      C.GoString(info.op_name),
				Name:    C.GoString(info.tensor_name),
				DType:   C.GoString(info.dtype_name),
				Backend: C.GoString(info.backend_name),
				TEnd:    int64(elapsed),
			}

			for d := 0; d < 4; d++ {
				ev.OutShape = append(ev.OutShape, int64(info.shape[d]))
			}
			for k := 0; k < int(info.n_srcs); k++ {
				if info.src_names[k] != nil {
					ev.SrcNames = append(ev.SrcNames, C.GoString(info.src_names[k]))
				}
			}

			events = append(events, ev)
		}
	}

	cursor := passStartTime.UnixNano()
	for i := range events {
		events[i].TStart = cursor
		events[i].TEnd = cursor + events[i].TEnd
		cursor = events[i].TEnd
	}

	return events
}
