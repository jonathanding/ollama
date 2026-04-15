// cuda_overlap_bench.cu
//
// Measures whether GPU compute (Streaming Multiprocessor / SM) and
// Host-to-Device (H2D) DMA (Copy Engine) can run in parallel on this machine.
//
// This answers the key question for Phase 2 of the MoE async pipeline:
// "Can we hide PCIe transfer latency behind GPU computation?"
//
// Build:
//   mkdir build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release
//   cmake --build .
//   ./cuda_overlap_bench
//
// Reference hardware: RTX 3090, PCIe 4.0 x16, 128 GB DDR5

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d - status %d\n",             \
                    __FILE__, __LINE__, (int)_st);                              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Size of one layer's MoE expert weights: 996 MB
// (ffn_gate_exps 288 MB + ffn_up_exps 288 MB + ffn_down_exps 420 MB)
static const size_t LAYER_BYTES = 996ULL * 1024 * 1024;

// Number of pipeline layers to simulate (layers whose MoE experts are in CPU RAM)
static const int N_PIPELINE_LAYERS = 28;

// Number of resident layers (MoE experts fully in VRAM, no H2D copy needed)
static const int N_RESIDENT_LAYERS = 20;

// MoE FFN matrix dimensions for Qwen3-Coder-Next 80B:
//
//   gate / up projections:
//     input:  [batch=1024 tokens, K1=2048 embedding dim]
//     weight: [K1=2048, N1=5120]  (10 selected experts x expert_ffn_dim 512)
//     output: [1024, 5120]
//
//   down projection:
//     input:  [1024, K2=5120]  (= gate output elementwise-mul up output)
//     weight: [K2=5120, N2=2048]
//     output: [1024, 2048]     (back to embedding dim)
//
// NOTE: in the real model, ggml uses MUL_MAT_ID which groups tokens per expert,
// so the actual FLOPs are spread across 512 experts. The shapes above approximate
// the total work assuming ~99% expert activation during prefill (batch=1024 tokens,
// top-10 selection, 512 experts -> ~10240 activations covering nearly all experts).
static const int GEMM_M  = 1024;  // batch size (number of prefill tokens)
static const int GEMM_K1 = 2048;  // embedding dimension
static const int GEMM_N1 = 5120;  // gate/up output dim (10 experts x 512)
static const int GEMM_K2 = 5120;  // down input dim  (= N1)
static const int GEMM_N2 = 2048;  // down output dim (= K1)

// Per-layer GPU compute time measured in Phase 1 experiment (ms).
// Includes attention/SSM (~2.5ms), MoE FFN (~8.1ms), and ggml scheduling overhead.
// Used for prefill predictions since cublasSgemm alone cannot reproduce the full
// ggml scheduler overhead that dominates the real per-layer time.
static const float REAL_LAYER_COMPUTE_MS = 10.6f;

// Number of repetitions for each measurement phase
static const int BANDWIDTH_REPS = 10;
static const int GEMM_REPS      = 50;
static const int OVERLAP_REPS   = 28; // one per simulated pipeline layer

// ---------------------------------------------------------------------------
// Utility: elapsed time between two CUDA events in milliseconds
// ---------------------------------------------------------------------------
static float event_ms(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

// ---------------------------------------------------------------------------
// GPU device info
// ---------------------------------------------------------------------------
static void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("  Video RAM (VRAM):          %.0f MB\n",
           prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("  Streaming Multiprocessors: %d SMs\n", prop.multiProcessorCount);
    // asyncEngineCount > 0 means the device has a dedicated Copy Engine
    // that can run H2D transfers concurrently with SM computation
    printf("  Async Copy Engine count:   %d  (%s)\n",
           prop.asyncEngineCount,
           prop.asyncEngineCount > 0
               ? "H2D and SM compute CAN overlap in hardware"
               : "WARNING: no dedicated Copy Engine, overlap not possible");
    printf("\n");
}

// ---------------------------------------------------------------------------
// Phase 2: H2D bandwidth measurement (synchronous, for baseline)
// ---------------------------------------------------------------------------
static float measure_bandwidth(
    void * dst_vram,        // destination: VRAM staging buffer
    const void * src_host,  // source: host memory (pageable or pinned)
    size_t bytes,
    int reps,
    const char * label)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // warm-up run (not timed)
    CUDA_CHECK(cudaMemcpy(dst_vram, src_host, bytes, cudaMemcpyHostToDevice));

    float total_ms = 0.0f;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(dst_vram, src_host, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        total_ms += event_ms(t0, t1);
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    float avg_ms  = total_ms / reps;
    float gbps    = (float)bytes / 1e9f / (avg_ms / 1e3f);
    printf("  %-42s  %6.1f ms   %5.1f GB/s\n", label, avg_ms, gbps);
    return avg_ms;
}

// ---------------------------------------------------------------------------
// MoE FFN simulation: runs gate + up + down matrix multiplications
//
// Simulates a single layer's MoE FFN forward pass:
//   gate: [M, K1] x [K1, N1] -> [M, N1]   (gate projection)
//   up:   [M, K1] x [K1, N1] -> [M, N1]   (up projection, same shape as gate)
//   down: [M, K2] x [K2, N2] -> [M, N2]   (down projection, K2=N1, N2=K1)
//
// Note: activation function (SiLU applied to gate * up) is omitted since
// it is memory-bandwidth-bound and small relative to the matrix multiplications.
//
// Buffer layout:
//   d_A: [M x K1]   token embeddings (input)
//   d_B: [K1 x N1]  weight matrix, reused for down (same total elements as [K2 x N2])
//   d_C: [M x N1]   gate/up output (intermediate), used as down input
//   d_D: [M x N2]   down output
// ---------------------------------------------------------------------------
static void run_moe_ffn(cublasHandle_t blas,
                         float * d_A, float * d_B, float * d_C, float * d_D)
{
    const float alpha = 1.0f, beta = 0.0f;

    // cublasSgemm uses column-major order.
    // For row-major C(M,N) = A(M,K) * B(K,N), the call is:
    //   cublasSgemm(N, M, K, B, ldb=N, A, lda=K, C, ldc=N)

    // gate: C = A * B_gate   [M, K1] x [K1, N1]
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             GEMM_N1, GEMM_M, GEMM_K1,
                             &alpha, d_B, GEMM_N1,
                                     d_A, GEMM_K1,
                             &beta,  d_C, GEMM_N1));

    // up: C = A * B_up   [M, K1] x [K1, N1]  (same shape, reuse buffers)
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             GEMM_N1, GEMM_M, GEMM_K1,
                             &alpha, d_B, GEMM_N1,
                                     d_A, GEMM_K1,
                             &beta,  d_C, GEMM_N1));

    // down: D = C * B_down   [M, K2] x [K2, N2]
    // B_down reuses d_B memory (K2*N2 = K1*N1 = same total elements),
    // treated as [K2=5120, N2=2048] in row-major i.e. [N2=2048, K2=5120] in col-major
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             GEMM_N2, GEMM_M, GEMM_K2,
                             &alpha, d_B, GEMM_N2,
                                     d_C, GEMM_K2,
                             &beta,  d_D, GEMM_N2));
}

// ---------------------------------------------------------------------------
// Phase 3: GPU compute time measurement
// ---------------------------------------------------------------------------
static float measure_gemm(cublasHandle_t blas,
                           float * d_A, float * d_B, float * d_C, float * d_D,
                           int reps)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // warm-up
    run_moe_ffn(blas, d_A, d_B, d_C, d_D);
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        run_moe_ffn(blas, d_A, d_B, d_C, d_D);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        total_ms += event_ms(t0, t1);
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    float avg_ms = total_ms / reps;
    printf("  gate [%dx%d]*[%dx%d] + up (same) + down [%dx%d]*[%dx%d]\n",
           GEMM_M, GEMM_K1, GEMM_K1, GEMM_N1,
           GEMM_M, GEMM_K2, GEMM_K2, GEMM_N2);
    printf("  benchmark GEMM (3x matmul):   %6.2f ms   (avg of %d runs)\n",
           avg_ms, reps);
    printf("  real per-layer compute time:  %6.2f ms   (from Phase 1 measurement,\n",
           REAL_LAYER_COMPUTE_MS);
    printf("                                            includes ggml scheduling overhead)\n");
    return avg_ms;
}

// ---------------------------------------------------------------------------
// Phase 4/5: overlap measurement
//
// How overlap is calculated:
//
//   Four CUDA events timestamp copy and compute independently on the GPU clock:
//     copy_start, copy_end  (on copy_stream)
//     gemm_start, gemm_end  (on compute_stream)
//
//   wall_ms  = span from earliest start to latest end
//            = max(copy_end, gemm_end) - min(copy_start, gemm_start)
//   serial   = copy_ms + gemm_ms  (hypothetical sequential execution)
//   saved    = serial - wall_ms
//   overlap% = saved / gemm_ms x 100   (clamped to [0, 100])
//
//   Pageable result (0%):
//     cudaMemcpyAsync with pageable source blocks the CPU thread internally
//     while CUDA runtime stages data through an internal pinned buffer.
//     Because the CPU is blocked, cublasSgemm is not submitted until after
//     the copy finishes -> gemm_start >= copy_end -> wall = serial -> 0% overlap.
//
//   Pinned result (100%):
//     cudaMemcpyAsync with pinned source returns immediately (true DMA).
//     cublasSgemm is submitted right after, both streams run concurrently.
//     Copy Engine (DMA) and SM (compute) are independent hardware units on RTX 3090.
//     wall = copy_ms (GEMM hidden inside copy duration) -> 100% overlap.
// ---------------------------------------------------------------------------
struct OverlapResult {
    float avg_copy_ms;
    float avg_gemm_ms;
    float avg_wall_ms;
    float avg_overlap_pct;
};

static OverlapResult measure_overlap(
    void *       buf_a,      // VRAM staging buffer A (SM reads from here)
    void *       buf_b,      // VRAM staging buffer B (Copy Engine writes here)
    const void * src_host,   // host source (pageable or pinned)
    float *      d_A,        // VRAM GEMM matrix A
    float *      d_B,        // VRAM GEMM weight matrix
    float *      d_C,        // VRAM GEMM intermediate
    float *      d_D,        // VRAM GEMM output
    int          reps,
    const char * label)
{
    cudaStream_t copy_stream, compute_stream;
    CUDA_CHECK(cudaStreamCreate(&copy_stream));
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    // Each stream needs its own cuBLAS handle bound to it
    cublasHandle_t blas_stream;
    CUBLAS_CHECK(cublasCreate(&blas_stream));
    CUBLAS_CHECK(cublasSetStream(blas_stream, compute_stream));

    cudaEvent_t copy_start, copy_end, gemm_start, gemm_end;
    CUDA_CHECK(cudaEventCreate(&copy_start));
    CUDA_CHECK(cudaEventCreate(&copy_end));
    CUDA_CHECK(cudaEventCreate(&gemm_start));
    CUDA_CHECK(cudaEventCreate(&gemm_end));

    // warm-up: one untimed iteration
    CUDA_CHECK(cudaMemcpyAsync(buf_b, src_host, LAYER_BYTES,
                               cudaMemcpyHostToDevice, copy_stream));
    run_moe_ffn(blas_stream, d_A, d_B, d_C, d_D);
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    float sum_copy = 0, sum_gemm = 0, sum_wall = 0, sum_overlap = 0;

    printf("  %s (%d layers)\n", label, reps);

    void * cur_buf_a = buf_a;
    void * cur_buf_b = buf_b;

    for (int i = 0; i < reps; i++) {
        // --- launch copy and compute concurrently ---
        CUDA_CHECK(cudaEventRecord(copy_start, copy_stream));
        CUDA_CHECK(cudaMemcpyAsync(cur_buf_b, src_host, LAYER_BYTES,
                                   cudaMemcpyHostToDevice, copy_stream));
        CUDA_CHECK(cudaEventRecord(copy_end, copy_stream));

        CUDA_CHECK(cudaEventRecord(gemm_start, compute_stream));
        run_moe_ffn(blas_stream, d_A, d_B, d_C, d_D);
        CUDA_CHECK(cudaEventRecord(gemm_end, compute_stream));

        // wait for both to finish
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // --- compute wall time ---
        // All four events share the GPU clock (same device), so cross-stream
        // elapsed time is valid. We find which started first, then compute:
        //   wall = max(copy_end, gemm_end) - min(copy_start, gemm_start)
        float copy_ms = event_ms(copy_start, copy_end);
        float gemm_ms = event_ms(gemm_start, gemm_end);

        // gemm_start_offset > 0 means copy_start came before gemm_start
        float gemm_start_offset = event_ms(copy_start, gemm_start);
        float wall_ms;
        if (gemm_start_offset >= 0.0f) {
            // copy started first; wall ends at the later of copy_end or gemm_end
            wall_ms = std::max(copy_ms, gemm_start_offset + gemm_ms);
        } else {
            // compute started first
            float copy_start_offset = -gemm_start_offset;
            wall_ms = std::max(gemm_ms, copy_start_offset + copy_ms);
        }

        // --- compute overlap percentage ---
        // overlap = (serial - wall) / gemm_ms, clamped to [0, 100]
        // gemm_ms is the denominator because it represents the maximum possible
        // saving (best case: GEMM is fully hidden inside the copy duration)
        float serial_ms   = copy_ms + gemm_ms;
        float saved_ms    = serial_ms - wall_ms;
        float overlap_pct = std::max(0.0f,
                            std::min(100.0f, saved_ms / gemm_ms * 100.0f));

        printf("    layer %2d: copy=%6.1fms  gemm=%5.1fms  wall=%6.1fms"
               "  overlap=%5.1f%%\n",
               i, copy_ms, gemm_ms, wall_ms, overlap_pct);

        sum_copy    += copy_ms;
        sum_gemm    += gemm_ms;
        sum_wall    += wall_ms;
        sum_overlap += overlap_pct;

        // double-buffer flip: A <-> B
        void * tmp = cur_buf_a;
        cur_buf_a  = cur_buf_b;
        cur_buf_b  = tmp;
    }

    CUDA_CHECK(cudaEventDestroy(copy_start));
    CUDA_CHECK(cudaEventDestroy(copy_end));
    CUDA_CHECK(cudaEventDestroy(gemm_start));
    CUDA_CHECK(cudaEventDestroy(gemm_end));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUBLAS_CHECK(cublasDestroy(blas_stream));

    OverlapResult r;
    r.avg_copy_ms     = sum_copy    / reps;
    r.avg_gemm_ms     = sum_gemm    / reps;
    r.avg_wall_ms     = sum_wall    / reps;
    r.avg_overlap_pct = sum_overlap / reps;

    printf("    avg: copy=%6.1fms  gemm=%5.1fms  wall=%6.1fms  overlap=%5.1f%%\n\n",
           r.avg_copy_ms, r.avg_gemm_ms, r.avg_wall_ms, r.avg_overlap_pct);

    return r;
}

// ---------------------------------------------------------------------------
// Verdict helpers
// ---------------------------------------------------------------------------
static const char * overlap_verdict(float pct) {
    if (pct >= 60.0f) return "GOOD    - Copy Engine and SM overlap effectively";
    if (pct >= 30.0f) return "PARTIAL - some overlap; pinned source would help";
    return               "POOR    - little overlap; pageable source is blocking";
}

// Prefill time prediction using Phase 1 measured compute time.
//
// Phase 2 double-buffering means each pipeline layer costs:
//   max(copy_time, compute_time)   instead of   copy_time + compute_time
//
// We use REAL_LAYER_COMPUTE_MS (from Phase 1) rather than the benchmark
// GEMM time, because the real per-layer compute includes ggml scheduling
// overhead that cublasSgemm does not reproduce.
static void print_prefill_prediction(
    const char * label,
    float avg_copy_ms,
    float avg_wall_ms,
    int   n_resident,
    int   n_pipeline)
{
    // Resident layers: no H2D copy, just GPU compute
    float resident_ms = n_resident * REAL_LAYER_COMPUTE_MS;

    // Pipeline layers serial baseline (current Phase 1 behavior):
    //   each layer = copy (pageable, synchronous) + compute
    // For serial prediction we use the copy time from the source being tested
    // plus the real compute time.
    float serial_per_layer  = avg_copy_ms + REAL_LAYER_COMPUTE_MS;
    float serial_total_ms   = n_pipeline * serial_per_layer;

    // Pipeline layers with Phase 2 double-buffering:
    //   each layer = max(copy, compute)  [they overlap]
    // avg_wall_ms from the benchmark already measures max(copy, benchmark_gemm).
    // Since real compute (10.6ms) may differ from benchmark GEMM, recalculate:
    float parallel_per_layer = std::max(avg_copy_ms, REAL_LAYER_COMPUTE_MS);
    float parallel_total_ms  = n_pipeline * parallel_per_layer;

    float total_serial   = resident_ms + serial_total_ms;
    float total_parallel = resident_ms + parallel_total_ms;
    float saved_pct = (total_serial - total_parallel) / total_serial * 100.0f;

    printf("  %s\n", label);
    printf("    resident layers (%2d x %.1fms):          %6.0f ms\n",
           n_resident, REAL_LAYER_COMPUTE_MS, resident_ms);
    printf("    pipeline layers (%2d) serial  (%.1f+%.1f)ms: %6.0f ms\n",
           n_pipeline, avg_copy_ms, REAL_LAYER_COMPUTE_MS, serial_total_ms);
    printf("    pipeline layers (%2d) parallel max(%.1f,%.1f)ms: %6.0f ms\n",
           n_pipeline, avg_copy_ms, REAL_LAYER_COMPUTE_MS, parallel_total_ms);
    printf("    predicted total serial:   %6.0f ms\n", total_serial);
    printf("    predicted total parallel: %6.0f ms\n", total_parallel);
    printf("    predicted saving: %.1f%%\n\n", saved_pct);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("========================================\n");
    printf("CUDA Overlap Benchmark\n");
    printf("(Phase 2 MoE async pipeline feasibility)\n");
    printf("========================================\n\n");

    print_device_info();

    // --- allocate host buffers -------------------------------------------

    printf("[Phase 1] Allocating and warming up host buffers\n");
    printf("  Buffer size per layer: %zu MB\n\n", LAYER_BYTES / 1024 / 1024);

    // pageable host buffer (simulates mmap-loaded model weights after warm-up)
    void * h_pageable = malloc(LAYER_BYTES);
    if (!h_pageable) { fprintf(stderr, "malloc failed\n"); return 1; }
    // Touch every page to ensure physical pages are resident (simulates
    // the steady-state after the model has been running for a while)
    memset(h_pageable, 1, LAYER_BYTES);
    printf("  pageable host buffer: %zu MB, all pages touched (warm-up done)\n",
           LAYER_BYTES / 1024 / 1024);

    // pinned host buffer (locked in physical RAM, Copy Engine can DMA directly)
    void * h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, LAYER_BYTES));
    memset(h_pinned, 1, LAYER_BYTES);
    printf("  pinned host buffer:   %zu MB, cudaMallocHost succeeded\n\n",
           LAYER_BYTES / 1024 / 1024);

    // --- allocate VRAM buffers -------------------------------------------

    // Two staging buffers for double-buffering (Buffer A and Buffer B)
    void * d_buf_a = nullptr;
    void * d_buf_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf_a, LAYER_BYTES));
    CUDA_CHECK(cudaMalloc(&d_buf_b, LAYER_BYTES));

    // GEMM matrices for MoE FFN simulation:
    //   d_gemm_A: [M x K1]  token embeddings
    //   d_gemm_B: [K1 x N1] weight matrix, also reused for down [K2 x N2]
    //             (K1*N1 == K2*N2 == 2048*5120, same buffer is safe)
    //   d_gemm_C: [M x N1]  gate/up output (intermediate)
    //   d_gemm_D: [M x N2]  down output
    float * d_gemm_A = nullptr;
    float * d_gemm_B = nullptr;
    float * d_gemm_C = nullptr;
    float * d_gemm_D = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gemm_A, (size_t)GEMM_M  * GEMM_K1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm_B, (size_t)GEMM_K1 * GEMM_N1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm_C, (size_t)GEMM_M  * GEMM_N1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm_D, (size_t)GEMM_M  * GEMM_N2 * sizeof(float)));

    // Fill with non-zero values to avoid zero-shortcut paths in cuBLAS
    CUDA_CHECK(cudaMemset(d_gemm_A, 1, (size_t)GEMM_M  * GEMM_K1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gemm_B, 1, (size_t)GEMM_K1 * GEMM_N1 * sizeof(float)));

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));

    // --- Phase 2: bandwidth ---------------------------------------------

    printf("[Phase 2] H2D Bandwidth (Host-to-Device, synchronous baseline)\n");
    float bw_pageable_ms = measure_bandwidth(d_buf_a, h_pageable,
                                             LAYER_BYTES, BANDWIDTH_REPS,
                                             "pageable (mmap-like) -> VRAM");
    float bw_pinned_ms   = measure_bandwidth(d_buf_a, h_pinned,
                                             LAYER_BYTES, BANDWIDTH_REPS,
                                             "pinned (cudaMallocHost) -> VRAM");
    printf("\n");

    // --- Phase 3: GPU compute time --------------------------------------

    printf("[Phase 3] GPU Compute Time (MoE FFN simulation: gate + up + down)\n");
    float gemm_ms = measure_gemm(blas, d_gemm_A, d_gemm_B, d_gemm_C, d_gemm_D,
                                 GEMM_REPS);
    printf("\n");

    // --- Phase 4: overlap with pageable source --------------------------

    printf("[Phase 4] Overlap Measurement - pageable source\n");
    OverlapResult ov_pageable = measure_overlap(
        d_buf_a, d_buf_b, h_pageable,
        d_gemm_A, d_gemm_B, d_gemm_C, d_gemm_D,
        OVERLAP_REPS,
        "pageable source");

    // --- Phase 5: overlap with pinned source ----------------------------

    printf("[Phase 5] Overlap Measurement - pinned source\n");
    OverlapResult ov_pinned = measure_overlap(
        d_buf_a, d_buf_b, h_pinned,
        d_gemm_A, d_gemm_B, d_gemm_C, d_gemm_D,
        OVERLAP_REPS,
        "pinned source");

    // --- Verdict --------------------------------------------------------

    printf("========================================\n");
    printf("VERDICT\n");
    printf("========================================\n\n");

    printf("  pageable overlap: %5.1f%%  -> %s\n",
           ov_pageable.avg_overlap_pct, overlap_verdict(ov_pageable.avg_overlap_pct));
    printf("  pinned   overlap: %5.1f%%  -> %s\n\n",
           ov_pinned.avg_overlap_pct, overlap_verdict(ov_pinned.avg_overlap_pct));

    printf("Prefill time prediction (1024 tokens, %d resident + %d pipeline layers)\n",
           N_RESIDENT_LAYERS, N_PIPELINE_LAYERS);
    printf("Using Phase 1 measured compute time: %.1f ms/layer\n\n",
           REAL_LAYER_COMPUTE_MS);

    print_prefill_prediction(
        "pageable source (current Phase 1 state, no change):",
        bw_pageable_ms, ov_pageable.avg_wall_ms,
        N_RESIDENT_LAYERS, N_PIPELINE_LAYERS);

    print_prefill_prediction(
        "pinned source + Phase 2 double-buffering:",
        bw_pinned_ms, ov_pinned.avg_wall_ms,
        N_RESIDENT_LAYERS, N_PIPELINE_LAYERS);

    printf("NOTE: predictions assume ideal scheduling.\n");
    printf("Actual gains will be lower due to stream sync overhead (~0.1-0.2ms/layer).\n");

    // --- cleanup --------------------------------------------------------

    CUBLAS_CHECK(cublasDestroy(blas));
    CUDA_CHECK(cudaFree(d_buf_a));
    CUDA_CHECK(cudaFree(d_buf_b));
    CUDA_CHECK(cudaFree(d_gemm_A));
    CUDA_CHECK(cudaFree(d_gemm_B));
    CUDA_CHECK(cudaFree(d_gemm_C));
    CUDA_CHECK(cudaFree(d_gemm_D));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    free(h_pageable);

    return 0;
}
