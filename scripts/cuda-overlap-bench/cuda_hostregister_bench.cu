// cuda_hostregister_bench.cu
//
// Tests whether cudaHostRegister() can be used as a lightweight alternative
// to cudaMallocHost() for making MoE expert weights available for direct DMA.
//
// Background:
//   Ollama loads model weights via mmap(). These are pageable memory pages,
//   which force CUDA to use an internal CPU-side staging step during H2D copy,
//   blocking the CPU thread and preventing overlap with GPU compute.
//
//   cudaHostRegister() can "register" existing memory as pinned without
//   reallocating it, potentially giving us pinned-memory DMA speed at zero
//   extra memory cost. However, Windows has restrictions on which memory
//   types can be registered.
//
// This benchmark tests three sources:
//   A. malloc()          + cudaHostRegister  (baseline: simplest case)
//   B. Windows mmap()    + cudaHostRegister  (actual Ollama model loading path)
//   C. cudaMallocHost()                      (reference: best possible pinned)
//
// For each registered source, we measure:
//   1. Whether cudaHostRegister() succeeds at all
//   2. Synchronous H2D bandwidth
//   3. Overlap percentage between H2D copy and GPU compute (same methodology
//      as cuda_overlap_bench.cu)
//
// Build: same CMakeLists.txt, add cuda_hostregister_bench as a second target.
//
// Reference hardware: RTX 3090, Windows 11, PCIe 4.0 x16

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Windows-specific headers for mmap simulation
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d - status %d\n",             \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Configuration (must match cuda_overlap_bench.cu)
// ---------------------------------------------------------------------------

static const size_t LAYER_BYTES = 996ULL * 1024 * 1024;

static const int GEMM_M  = 1024;
static const int GEMM_K1 = 2048;
static const int GEMM_N1 = 5120;
static const int GEMM_K2 = 5120;
static const int GEMM_N2 = 2048;

static const float REAL_LAYER_COMPUTE_MS = 10.6f;

static const int BANDWIDTH_REPS = 10;
static const int OVERLAP_REPS   = 28;

// Temporary file path for mmap simulation
#ifdef _WIN32
static const char * TEMP_FILE_PATH = "C:\\Windows\\Temp\\cuda_hostregister_bench_tmp.bin";
#else
static const char * TEMP_FILE_PATH = "/tmp/cuda_hostregister_bench_tmp.bin";
#endif

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

static float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// ---------------------------------------------------------------------------
// MoE FFN simulation (gate + up + down, same as cuda_overlap_bench.cu)
// ---------------------------------------------------------------------------

static void run_moe_ffn(cublasHandle_t blas,
                        float * d_A, float * d_B, float * d_C, float * d_D)
{
    const float alpha = 1.0f, beta = 0.0f;

    // gate: [M, K1] x [K1, N1]
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             GEMM_N1, GEMM_M, GEMM_K1,
                             &alpha, d_B, GEMM_N1, d_A, GEMM_K1,
                             &beta,  d_C, GEMM_N1));
    // up: same shape
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             GEMM_N1, GEMM_M, GEMM_K1,
                             &alpha, d_B, GEMM_N1, d_A, GEMM_K1,
                             &beta,  d_C, GEMM_N1));
    // down: [M, K2] x [K2, N2]
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                             GEMM_N2, GEMM_M, GEMM_K2,
                             &alpha, d_B, GEMM_N2, d_C, GEMM_K2,
                             &beta,  d_D, GEMM_N2));
}

// ---------------------------------------------------------------------------
// Windows mmap: create a temporary file and map it into virtual memory.
// Returns the mapped address (page-aligned), or nullptr on failure.
// The caller must unmap with unmap_file() and delete the file when done.
// ---------------------------------------------------------------------------

#ifdef _WIN32

struct MappedFile {
    void *   addr;
    size_t   size;
    HANDLE   file_handle;
    HANDLE   mapping_handle;
};

static MappedFile map_file(const char * path, size_t size) {
    MappedFile m = {nullptr, 0, INVALID_HANDLE_VALUE, nullptr};

    // Create (or overwrite) the temporary file
    m.file_handle = CreateFileA(
        path,
        GENERIC_READ | GENERIC_WRITE,
        0,               // no sharing
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
        nullptr);

    if (m.file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "  CreateFileA failed: error %lu\n", GetLastError());
        return m;
    }

    // Set file size
    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)size;
    if (!SetFilePointerEx(m.file_handle, li, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(m.file_handle)) {
        fprintf(stderr, "  SetFilePointerEx/SetEndOfFile failed: error %lu\n",
                GetLastError());
        CloseHandle(m.file_handle);
        m.file_handle = INVALID_HANDLE_VALUE;
        return m;
    }

    // Create a file mapping object (read/write, no name)
    m.mapping_handle = CreateFileMappingA(
        m.file_handle,
        nullptr,
        PAGE_READWRITE,
        (DWORD)(size >> 32),
        (DWORD)(size & 0xFFFFFFFF),
        nullptr);

    if (!m.mapping_handle) {
        fprintf(stderr, "  CreateFileMappingA failed: error %lu\n", GetLastError());
        CloseHandle(m.file_handle);
        m.file_handle = INVALID_HANDLE_VALUE;
        return m;
    }

    // Map the entire file into virtual address space
    m.addr = MapViewOfFile(
        m.mapping_handle,
        FILE_MAP_READ | FILE_MAP_WRITE,
        0, 0,   // offset = 0
        size);

    if (!m.addr) {
        fprintf(stderr, "  MapViewOfFile failed: error %lu\n", GetLastError());
        CloseHandle(m.mapping_handle);
        CloseHandle(m.file_handle);
        m.mapping_handle = nullptr;
        m.file_handle    = INVALID_HANDLE_VALUE;
        return m;
    }

    m.size = size;
    return m;
}

static void unmap_file(MappedFile & m) {
    if (m.addr)            { UnmapViewOfFile(m.addr);     m.addr = nullptr; }
    if (m.mapping_handle)  { CloseHandle(m.mapping_handle); m.mapping_handle = nullptr; }
    if (m.file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(m.file_handle);   // FILE_FLAG_DELETE_ON_CLOSE handles deletion
        m.file_handle = INVALID_HANDLE_VALUE;
    }
}

#else // POSIX fallback (Linux / macOS)

struct MappedFile {
    void * addr;
    size_t size;
    int    fd;
};

static MappedFile map_file(const char * path, size_t size) {
    MappedFile m = {nullptr, 0, -1};
    m.fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (m.fd < 0) { perror("open"); return m; }
    if (ftruncate(m.fd, (off_t)size) < 0) { perror("ftruncate"); close(m.fd); m.fd=-1; return m; }
    m.addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0);
    if (m.addr == MAP_FAILED) { perror("mmap"); close(m.fd); m.fd=-1; m.addr=nullptr; return m; }
    m.size = size;
    return m;
}

static void unmap_file(MappedFile & m) {
    if (m.addr) { munmap(m.addr, m.size); m.addr = nullptr; }
    if (m.fd >= 0) { close(m.fd); unlink(TEMP_FILE_PATH); m.fd = -1; }
}

#endif

// ---------------------------------------------------------------------------
// Try cudaHostRegister on a memory region.
// Returns true if successful, false otherwise (prints the CUDA error).
// On success, the memory is pinned until cudaHostUnregister() is called.
// ---------------------------------------------------------------------------

static bool try_host_register(void * ptr, size_t size, const char * label) {
    // cudaHostRegisterDefault (0): let CUDA choose the best registration type.
    // On Windows, this may fail for mmap regions depending on the memory type.
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        printf("  cudaHostRegister(%s): SUCCESS\n", label);
        return true;
    }
    // Do NOT call CUDA_CHECK here — we want to handle the error gracefully
    cudaGetLastError();  // clear the error state
    printf("  cudaHostRegister(%s): FAILED - %s\n", label, cudaGetErrorString(err));
    return false;
}

// ---------------------------------------------------------------------------
// Bandwidth measurement (synchronous H2D copy)
// ---------------------------------------------------------------------------

static float measure_bandwidth(void * d_dst, const void * h_src,
                                size_t bytes, int reps, const char * label)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // warm-up
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice));

    float total = 0;
    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        total += event_ms(t0, t1);
    }

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    float avg_ms = total / reps;
    float gbps   = (float)bytes / 1e9f / (avg_ms / 1e3f);
    printf("    H2D bandwidth %-35s %6.1f ms  %5.1f GB/s\n",
           label, avg_ms, gbps);
    return avg_ms;
}

// ---------------------------------------------------------------------------
// Overlap measurement (same logic as cuda_overlap_bench.cu)
// ---------------------------------------------------------------------------

struct OverlapResult {
    bool  registered;       // whether cudaHostRegister succeeded
    float avg_copy_ms;
    float avg_gemm_ms;
    float avg_wall_ms;
    float avg_overlap_pct;
};

static OverlapResult measure_overlap(
    void *  d_buf_a, void * d_buf_b,
    const void * src,
    float * d_A, float * d_B, float * d_C, float * d_D,
    int reps)
{
    OverlapResult r = {};

    cudaStream_t copy_s, compute_s;
    CUDA_CHECK(cudaStreamCreate(&copy_s));
    CUDA_CHECK(cudaStreamCreate(&compute_s));

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, compute_s));

    cudaEvent_t cs, ce, gs, ge;
    CUDA_CHECK(cudaEventCreate(&cs));
    CUDA_CHECK(cudaEventCreate(&ce));
    CUDA_CHECK(cudaEventCreate(&gs));
    CUDA_CHECK(cudaEventCreate(&ge));

    // warm-up
    CUDA_CHECK(cudaMemcpyAsync(d_buf_b, src, LAYER_BYTES,
                               cudaMemcpyHostToDevice, copy_s));
    run_moe_ffn(blas, d_A, d_B, d_C, d_D);
    CUDA_CHECK(cudaStreamSynchronize(copy_s));
    CUDA_CHECK(cudaStreamSynchronize(compute_s));

    void * cur_a = d_buf_a;
    void * cur_b = d_buf_b;

    float sum_copy = 0, sum_gemm = 0, sum_wall = 0, sum_ov = 0;

    for (int i = 0; i < reps; i++) {
        CUDA_CHECK(cudaEventRecord(cs, copy_s));
        CUDA_CHECK(cudaMemcpyAsync(cur_b, src, LAYER_BYTES,
                                   cudaMemcpyHostToDevice, copy_s));
        CUDA_CHECK(cudaEventRecord(ce, copy_s));

        CUDA_CHECK(cudaEventRecord(gs, compute_s));
        run_moe_ffn(blas, d_A, d_B, d_C, d_D);
        CUDA_CHECK(cudaEventRecord(ge, compute_s));

        CUDA_CHECK(cudaStreamSynchronize(copy_s));
        CUDA_CHECK(cudaStreamSynchronize(compute_s));

        float copy_ms   = event_ms(cs, ce);
        float gemm_ms   = event_ms(gs, ge);
        float gs_offset = event_ms(cs, gs);  // positive = copy started first
        float wall_ms;
        if (gs_offset >= 0.0f)
            wall_ms = std::max(copy_ms, gs_offset + gemm_ms);
        else
            wall_ms = std::max(gemm_ms, -gs_offset + copy_ms);

        float serial  = copy_ms + gemm_ms;
        float overlap = std::max(0.0f,
                        std::min(100.0f,
                                 (serial - wall_ms) / gemm_ms * 100.0f));

        printf("    layer %2d: copy=%6.1fms  gemm=%5.1fms  wall=%6.1fms"
               "  overlap=%5.1f%%\n",
               i, copy_ms, gemm_ms, wall_ms, overlap);

        sum_copy += copy_ms;
        sum_gemm += gemm_ms;
        sum_wall += wall_ms;
        sum_ov   += overlap;

        void * tmp = cur_a; cur_a = cur_b; cur_b = tmp;
    }

    printf("    avg: copy=%6.1fms  gemm=%5.1fms  wall=%6.1fms  overlap=%5.1f%%\n",
           sum_copy/reps, sum_gemm/reps, sum_wall/reps, sum_ov/reps);

    CUDA_CHECK(cudaEventDestroy(cs)); CUDA_CHECK(cudaEventDestroy(ce));
    CUDA_CHECK(cudaEventDestroy(gs)); CUDA_CHECK(cudaEventDestroy(ge));
    CUDA_CHECK(cudaStreamDestroy(copy_s));
    CUDA_CHECK(cudaStreamDestroy(compute_s));
    CUBLAS_CHECK(cublasDestroy(blas));

    r.registered      = true;
    r.avg_copy_ms     = sum_copy / reps;
    r.avg_gemm_ms     = sum_gemm / reps;
    r.avg_wall_ms     = sum_wall / reps;
    r.avg_overlap_pct = sum_ov   / reps;
    return r;
}

// ---------------------------------------------------------------------------
// Print prefill prediction for one source
// ---------------------------------------------------------------------------
static void print_prediction(const char * label, float copy_ms,
                              int n_resident, int n_pipeline)
{
    float resident  = n_resident  * REAL_LAYER_COMPUTE_MS;
    float serial    = n_pipeline  * (copy_ms + REAL_LAYER_COMPUTE_MS);
    float parallel  = n_pipeline  * std::max(copy_ms, REAL_LAYER_COMPUTE_MS);
    float total_s   = resident + serial;
    float total_p   = resident + parallel;
    float saving    = (total_s - total_p) / total_s * 100.0f;

    printf("  %s\n", label);
    printf("    serial   total: %6.0f ms\n", total_s);
    printf("    parallel total: %6.0f ms  (predicted with Phase 2)\n", total_p);
    printf("    predicted saving: %.1f%%\n\n", saving);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("========================================\n");
    printf("cudaHostRegister Feasibility Benchmark\n");
    printf("(Phase 2 MoE: lightweight pinned memory path)\n");
    printf("========================================\n\n");

    // --- VRAM buffers ---
    void * d_buf_a; CUDA_CHECK(cudaMalloc(&d_buf_a, LAYER_BYTES));
    void * d_buf_b; CUDA_CHECK(cudaMalloc(&d_buf_b, LAYER_BYTES));

    float * d_A; CUDA_CHECK(cudaMalloc(&d_A, (size_t)GEMM_M  * GEMM_K1 * sizeof(float)));
    float * d_B; CUDA_CHECK(cudaMalloc(&d_B, (size_t)GEMM_K1 * GEMM_N1 * sizeof(float)));
    float * d_C; CUDA_CHECK(cudaMalloc(&d_C, (size_t)GEMM_M  * GEMM_N1 * sizeof(float)));
    float * d_D; CUDA_CHECK(cudaMalloc(&d_D, (size_t)GEMM_M  * GEMM_N2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_A, 1, (size_t)GEMM_M  * GEMM_K1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 1, (size_t)GEMM_K1 * GEMM_N1 * sizeof(float)));

    // --- Reference: cudaMallocHost (best-case pinned) ---
    printf("========================================\n");
    printf("[Reference] cudaMallocHost (best-case pinned)\n");
    printf("========================================\n");
    void * h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, LAYER_BYTES));
    memset(h_pinned, 1, LAYER_BYTES);
    measure_bandwidth(d_buf_a, h_pinned, LAYER_BYTES, BANDWIDTH_REPS,
                      "cudaMallocHost -> VRAM");
    printf("  Overlap measurement:\n");
    OverlapResult ref = measure_overlap(d_buf_a, d_buf_b, h_pinned,
                                        d_A, d_B, d_C, d_D, OVERLAP_REPS);
    printf("\n");

    // =========================================================
    // Source A: malloc + cudaHostRegister
    // =========================================================
    printf("========================================\n");
    printf("[Source A] malloc() + cudaHostRegister\n");
    printf("========================================\n");

    void * h_malloc = malloc(LAYER_BYTES);
    if (!h_malloc) { fprintf(stderr, "malloc failed\n"); return 1; }
    memset(h_malloc, 1, LAYER_BYTES);  // touch all pages (warm-up)
    printf("  malloc: %zu MB allocated and touched\n", LAYER_BYTES / 1024 / 1024);

    bool reg_malloc = try_host_register(h_malloc, LAYER_BYTES, "malloc");

    if (reg_malloc) {
        printf("  Bandwidth after registration:\n");
        float bw_a = measure_bandwidth(d_buf_a, h_malloc, LAYER_BYTES,
                                       BANDWIDTH_REPS, "malloc+registered -> VRAM");
        printf("  Overlap measurement:\n");
        OverlapResult ov_a = measure_overlap(d_buf_a, d_buf_b, h_malloc,
                                             d_A, d_B, d_C, d_D, OVERLAP_REPS);
        printf("  Overlap: %.1f%%  (reference pinned: %.1f%%)\n\n",
               ov_a.avg_overlap_pct, ref.avg_overlap_pct);
    } else {
        printf("  Skipping bandwidth and overlap tests (registration failed).\n\n");
    }

    if (reg_malloc) cudaHostUnregister(h_malloc);
    free(h_malloc);

    // =========================================================
    // Source B: Windows mmap + cudaHostRegister
    // =========================================================
    printf("========================================\n");
    printf("[Source B] Windows mmap() + cudaHostRegister\n");
    printf("(simulates Ollama model weight loading path)\n");
    printf("========================================\n");
    printf("  Creating temporary file: %s\n", TEMP_FILE_PATH);

    MappedFile mf = map_file(TEMP_FILE_PATH, LAYER_BYTES);

    if (!mf.addr) {
        printf("  mmap creation FAILED - cannot test Source B.\n\n");
    } else {
        printf("  mmap: %zu MB mapped successfully\n", LAYER_BYTES / 1024 / 1024);
        // Touch all pages to simulate steady-state (model already loaded)
        memset(mf.addr, 1, LAYER_BYTES);
        printf("  All pages touched (warm-up done)\n");

        bool reg_mmap = try_host_register(mf.addr, LAYER_BYTES, "mmap");

        if (reg_mmap) {
            printf("  Bandwidth after registration:\n");
            float bw_b = measure_bandwidth(d_buf_a, mf.addr, LAYER_BYTES,
                                           BANDWIDTH_REPS, "mmap+registered -> VRAM");
            printf("  Overlap measurement:\n");
            OverlapResult ov_b = measure_overlap(d_buf_a, d_buf_b, mf.addr,
                                                 d_A, d_B, d_C, d_D, OVERLAP_REPS);
            printf("  Overlap: %.1f%%  (reference pinned: %.1f%%)\n\n",
                   ov_b.avg_overlap_pct, ref.avg_overlap_pct);
        } else {
            printf("  Skipping bandwidth and overlap tests (registration failed).\n");
            printf("  CONCLUSION: cudaHostRegister cannot be used on mmap memory.\n");
            printf("  Alternative path required (see VERDICT below).\n\n");
        }

        if (reg_mmap) cudaHostUnregister(mf.addr);
        unmap_file(mf);
    }

    // =========================================================
    // VERDICT
    // =========================================================
    printf("========================================\n");
    printf("VERDICT\n");
    printf("========================================\n\n");
    printf("Reference (cudaMallocHost): overlap=%.1f%%  copy=%.1fms\n\n",
           ref.avg_overlap_pct, ref.avg_copy_ms);
    printf("Prefill predictions (20 resident + 28 pipeline layers,\n");
    printf("per-layer compute = %.1f ms from Phase 1):\n\n",
           REAL_LAYER_COMPUTE_MS);
    print_prediction("cudaMallocHost (reference best case):",
                     ref.avg_copy_ms, 20, 28);
    printf("NOTE: predictions for Sources A/B printed inline above if\n");
    printf("registration succeeded.\n\n");
    printf("If Source B (mmap) registration FAILED:\n");
    printf("  -> cudaHostRegister is not viable for Ollama's model loading path.\n");
    printf("  -> Phase 2 must use an alternative pinned strategy:\n");
    printf("     Option 1: CPU-side pinned staging buffer (2x ~1GB cudaMallocHost)\n");
    printf("               CPU memcpy from mmap -> pinned staging, then DMA to VRAM\n");
    printf("               Cost: only 2 GB pinned (vs 32 GB for full weight pinning)\n");
    printf("     Option 2: Change Ollama model loading to use cudaMallocHost\n");
    printf("               for CPU-MoE layers (high memory cost, ~28 GB pinned)\n");

    // --- cleanup ---
    CUDA_CHECK(cudaFree(d_buf_a)); CUDA_CHECK(cudaFree(d_buf_b));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C)); CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFreeHost(h_pinned));

    return 0;
}
