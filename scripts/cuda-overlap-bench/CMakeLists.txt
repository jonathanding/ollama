cmake_minimum_required(VERSION 3.18)
project(cuda_overlap_bench CUDA CXX)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Helper: apply common settings to a benchmark target
# ---------------------------------------------------------------------------
function(configure_bench target)
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "80;86"
    )
    target_link_libraries(${target}
        PRIVATE
            CUDA::cublas
            CUDA::cudart
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            -O3
        >
    )
endfunction()

# ---------------------------------------------------------------------------
# Benchmark 1: H2D / compute overlap feasibility
# ---------------------------------------------------------------------------
add_executable(cuda_overlap_bench cuda_overlap_bench.cu)
configure_bench(cuda_overlap_bench)

# ---------------------------------------------------------------------------
# Benchmark 2: cudaHostRegister feasibility (malloc vs mmap)
# ---------------------------------------------------------------------------
add_executable(cuda_hostregister_bench cuda_hostregister_bench.cu)
configure_bench(cuda_hostregister_bench)

# On Windows, link against the system kernel library needed for
# CreateFileMapping / MapViewOfFile used in Source B (mmap simulation)
if(WIN32)
    target_link_libraries(cuda_hostregister_bench PRIVATE kernel32)
endif()

# ---------------------------------------------------------------------------
# Print build config summary
# ---------------------------------------------------------------------------
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures:   80 (Ampere A100), 86 (RTX 3090/3080)")
message(STATUS "Targets: cuda_overlap_bench, cuda_hostregister_bench")
