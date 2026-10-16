// task1/cuda/gemm_shared.cu
// ---------------------------------------------------------------------------
// Task 1 - 1.4 Shared Memory Tiling
//
// 数据流：Global Memory --(load)--> Shared Memory --(reuse)--> Register --> Compute
//
// 每个 Block 负责 C 的一个 TILE x TILE 分块：
//   1. 协作把 A 的一块 (TILE x TILE) 与 B 的一块 (TILE x TILE) 搬进 shared memory；
//   2. __syncthreads() 保证"所有人都装好数据"；
//   3. 每个 thread 在自己的 TILE 内做 K 方向累加，A/B 都从 shared memory 读。
//
// 一个 TILE 内的数据被 TILE 个 thread 复用 -> Global Memory Traffic 降为原来的 1/TILE 量级。
//
// 编译：nvcc -O3 -arch=native -std=c++17 gemm_shared.cu -lcublas -o gemm_shared
// 运行：./gemm_shared > results/gemm_shared.csv
// ---------------------------------------------------------------------------
#include "gemm_bench.h"

template <int TILE>
__global__ void gemm_shared_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // ---- 搬运：越界补 0，保证不读到非法地址，也不影响累加结果 ----
        const int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        const int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();  // 所有人装好数据后才能继续

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // 下一轮要覆盖 As/Bs，必须先确认本轮的读已经完成
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// 对每个 TILE 跑完全部标准 Shape；variant 名里带上 TILE，方便同一张 CSV 里对比
// ---------------------------------------------------------------------------
template <int TILE>
void run_tile_variant(cublasHandle_t handle) {
    const std::string variant = "shared-T" + std::to_string(TILE);
    const dim3 block(TILE, TILE);

    for (const GemmShape& shape : standard_gemm_shapes()) {
        const std::size_t a_bytes = static_cast<std::size_t>(shape.M) * shape.K * sizeof(float);
        const std::size_t b_bytes = static_cast<std::size_t>(shape.K) * shape.N * sizeof(float);
        const std::size_t c_bytes = static_cast<std::size_t>(shape.M) * shape.N * sizeof(float);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, a_bytes));
        CUDA_CHECK(cudaMalloc(&dB, b_bytes));
        CUDA_CHECK(cudaMalloc(&dC, c_bytes));
        init_gemm_inputs(dA, dB, shape.M, shape.N, shape.K);

        const dim3 grid((shape.N + TILE - 1) / TILE, (shape.M + TILE - 1) / TILE);
        auto launch = [&] {
            gemm_shared_kernel<TILE><<<grid, block>>>(dA, dB, dC, shape.M, shape.N, shape.K);
        };

        bench_gemm_case(handle, variant, shape, dA, dB, dC, launch);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
}

int main() {
    const DeviceInfo dev = query_device();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::printf("# task1 gemm_shared benchmark (shared memory tiling)\n");
    std::printf("# device   : %s (sm_%d%d, shared mem/block=%d B)\n", dev.name.c_str(),
                dev.cc_major, dev.cc_minor, dev.shared_mem_per_block);
    std::printf("# tile sizes: 8, 16, 32\n");
    std::printf("# baseline : cuBLAS SGEMM, CUBLAS_PEDANTIC_MATH, FP32\n");
    std::printf("# measurement: warmup=%d, iters=%d, CUDA Event, kernel-only\n",
                kWarmupIters, kMeasureIters);
    print_gemm_csv_header();
    std::fflush(stdout);

    run_tile_variant<8>(handle);
    run_tile_variant<16>(handle);
    run_tile_variant<32>(handle);

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
