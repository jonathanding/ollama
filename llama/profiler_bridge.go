package llama

/*
#include <stdlib.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

extern void GoProfilerCallback(uintptr_t handle, void* tensorPtr, _Bool ask);

static _Bool profilerEvalCallbackBridge(struct ggml_tensor* t, _Bool ask, void* user_data) {
    GoProfilerCallback((uintptr_t)user_data, (void*)t, ask);
    return 1; // always true: ask=true means "dispatch this node", ask=false means "continue"
}

// setProfilerEvalCallback sets the profiler bridge as eval callback.
// A separate C function is required because CGO cannot pass static function pointers directly.
static void setProfilerEvalCallback(struct llama_context* ctx, uintptr_t handle) {
    llama_context_set_eval_callback(ctx, profilerEvalCallbackBridge, (void*)handle);
}
*/
import "C"

import (
	"runtime/cgo"
	"time"
	"unsafe"

	"github.com/ollama/ollama/llm/profiler"
)

// SetEvalCallback registers a per-operator callback for inference profiling.
// Pass a non-nil TraceCollector to enable; nil to disable (zero overhead).
// Call after NewContextWithModel and before the first Decode.
func (c *Context) SetEvalCallback(col profiler.TraceCollector) {
	if c.profilerHandle != 0 {
		c.profilerHandle.Delete()
		c.profilerHandle = 0
	}
	if col == nil {
		C.llama_context_set_eval_callback(c.c, nil, nil)
		return
	}
	h := cgo.NewHandle(col)
	c.profilerHandle = h
	C.setProfilerEvalCallback(c.c, C.uintptr_t(uintptr(h)))
}

// GoProfilerCallback is called from the C eval callback bridge on the C thread.
// ask=true: tensor about to be dispatched. ask=false: tensor dispatched.
//
//export GoProfilerCallback
func GoProfilerCallback(handle C.uintptr_t, tensorPtr unsafe.Pointer, ask C.bool) {
	h := cgo.Handle(handle)
	col, ok := h.Value().(profiler.TraceCollector)
	if !ok {
		return
	}
	ptr := uintptr(tensorPtr)
	now := time.Now().UnixNano()
	if bool(ask) {
		col.RecordTensorStart(ptr, now)
		return
	}
	t := (*C.struct_ggml_tensor)(tensorPtr)
	col.RecordTensorEnd(ptr, extractTensorInfo(t), now)
}

// extractTensorInfo reads metadata from a C ggml_tensor pointer.
// Safe to call only during Decode() — tensor arena pointers are stable
// for the duration of a single llama_decode() call.
// This function must stay in this file (CGO file) since it accesses C structs.
func extractTensorInfo(t *C.struct_ggml_tensor) profiler.TensorInfo {
	info := profiler.TensorInfo{
		Op:   C.GoString(C.ggml_op_name(t.op)),
		Name: C.GoString(&t.name[0]),
	}

	// Output shape: ne[0..3]; trim trailing 1s
	shape := make([]int64, 4)
	for i := 0; i < 4; i++ {
		shape[i] = int64(t.ne[i])
	}
	for len(shape) > 1 && shape[len(shape)-1] == 1 {
		shape = shape[:len(shape)-1]
	}
	info.OutShape = shape

	// Data type name. CGO mangles C field "type" to "_type".
	info.DType = C.GoString(C.ggml_type_name(t._type))

	// Backend (device) where this tensor lives
	if t.buffer != nil {
		info.Backend = C.GoString(C.ggml_backend_buffer_name(t.buffer))
	}

	// Source tensor names → DAG edges.
	// GGML_MAX_SRC = 10; stop at first NULL.
	for i := 0; i < 10; i++ {
		src := t.src[i]
		if src == nil {
			break
		}
		info.SrcNames = append(info.SrcNames, C.GoString(&src.name[0]))
	}

	return info
}
