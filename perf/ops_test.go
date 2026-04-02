package perf

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestComputeFLOPs_MulMat(t *testing.T) {
	f := ComputeFLOPs("MUL_MAT", [][]int64{{4096, 4096}, {4096, 1}})
	assert.InDelta(t, 2*4096*4096*1, f, 1)
}

func TestComputeFLOPs_FlashAttn(t *testing.T) {
	f := ComputeFLOPs("FLASH_ATTN_EXT", [][]int64{{1, 32, 512, 128}, {1, 32, 512, 128}})
	assert.InDelta(t, 2.0*1*32*512*512*128, f, 1)
}

func TestComputeFLOPs_RMSNorm(t *testing.T) {
	f := ComputeFLOPs("RMS_NORM", [][]int64{{4096, 512}})
	assert.InDelta(t, 3*4096*512, f, 1)
}

func TestComputeFLOPs_Add(t *testing.T) {
	f := ComputeFLOPs("ADD", [][]int64{{4096}})
	assert.InDelta(t, 4096, f, 1)
}

func TestComputeFLOPs_View(t *testing.T) {
	f := ComputeFLOPs("VIEW", [][]int64{{4096}})
	assert.Equal(t, float64(0), f)
}

func TestComputeBytes_MulMat(t *testing.T) {
	b := ComputeBytes("MUL_MAT", [][]int64{{4096, 4096}, {4096, 1}}, "f32", "f16")
	expected := 2.0*4096*4096 + 4.0*4096*1 + 4.0*4096*1
	assert.InDelta(t, expected, b, 1)
}

func TestComputeBytes_MulMat_Q4_0(t *testing.T) {
	b := ComputeBytes("MUL_MAT", [][]int64{{4096, 4096}, {4096, 1}}, "f32", "q4_0")
	expected := 0.5625*4096*4096 + 4.0*4096*1 + 4.0*4096*1
	assert.InDelta(t, expected, b, 1)
}

func TestComputeBytes_RMSNorm(t *testing.T) {
	b := ComputeBytes("RMS_NORM", [][]int64{{4096, 512}}, "f32", "")
	expected := 2.0*4096*512*4 + 4096*4
	assert.InDelta(t, expected, b, 1)
}
