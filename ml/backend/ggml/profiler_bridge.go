package ggml

/*
#include <stdlib.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-backend.h"

extern void GoGGMLProfilerCallback(uintptr_t handle, void* tensorPtr, _Bool ask);

static _Bool ggmlProfilerEvalCallbackBridge(struct ggml_tensor* t, _Bool ask, void* user_data) {
    GoGGMLProfilerCallback((uintptr_t)user_data, (void*)t, ask);
    return 1;
}

static void setGGMLProfilerEvalCallback(ggml_backend_sched_t sched, uintptr_t handle) {
    ggml_backend_sched_set_eval_callback(sched, ggmlProfilerEvalCallbackBridge, (void*)handle);
}

static void clearGGMLProfilerEvalCallback(ggml_backend_sched_t sched) {
    ggml_backend_sched_set_eval_callback(sched, NULL, NULL);
}
*/
import "C"

import (
	"runtime/cgo"
	"time"
	"unsafe"

	"github.com/ollama/ollama/llm/profiler"
)

// SetEvalCallback registers a per-operator callback for inference profiling on the ggml backend scheduler.
// Pass a non-nil TraceCollector to enable; nil to disable.
func (b *Backend) SetEvalCallback(col profiler.TraceCollector) {
	if b.profilerHandle != 0 {
		b.profilerHandle.Delete()
		b.profilerHandle = 0
	}
	if col == nil {
		C.clearGGMLProfilerEvalCallback(b.sched)
		return
	}
	h := cgo.NewHandle(col)
	b.profilerHandle = h
	C.setGGMLProfilerEvalCallback(b.sched, C.uintptr_t(uintptr(h)))
}

// GoGGMLProfilerCallback is called from the C eval callback bridge.
// ask=true: tensor about to be dispatched. ask=false: tensor dispatched.
//
//export GoGGMLProfilerCallback
func GoGGMLProfilerCallback(handle C.uintptr_t, tensorPtr unsafe.Pointer, ask C.bool) {
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
	col.RecordTensorEnd(ptr, extractGGMLTensorInfo(t), now)
}

// extractGGMLTensorInfo reads metadata from a C ggml_tensor pointer.
func extractGGMLTensorInfo(t *C.struct_ggml_tensor) profiler.TensorInfo {
	info := profiler.TensorInfo{
		Op:   C.GoString(C.ggml_op_name(t.op)),
		Name: C.GoString(&t.name[0]),
	}

	shape := make([]int64, 4)
	for i := range 4 {
		shape[i] = int64(t.ne[i])
	}
	for len(shape) > 1 && shape[len(shape)-1] == 1 {
		shape = shape[:len(shape)-1]
	}
	info.OutShape = shape

	info.DType = C.GoString(C.ggml_type_name(t._type))

	if t.buffer != nil {
		info.Backend = C.GoString(C.ggml_backend_buffer_name(t.buffer))
	}

	for i := range 10 {
		src := t.src[i]
		if src == nil {
			break
		}
		info.SrcNames = append(info.SrcNames, C.GoString(&src.name[0]))
	}

	return info
}
