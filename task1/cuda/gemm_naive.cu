// task1/cuda/gemm_naive.cu
// ---------------------------------------------------------------------------
// Task 1 - 1.3 Naive GEMM：One Thread -> One C Element
//
//   C_{ij} = sum_k A_{ik} B_{kj}
//
// 这份实现"慢但容易看懂"，是后面 Shared Memory / Register Tiling 的起点。
// 它的性能瓶颈也很清楚：每个 thread 算一个 C 元素，K 次循环里
//   读 A 的一行（每次 4 Bytes，连续，可 coalesce）
//   读 B 的一列（每次跨 N*4 Bytes，**完全不 coalesce**）
// 所以 B 的每一次访问几乎都在浪费 cache line，这是它远慢于 cuBLAS 的主因之一。
//
// 编译：nvcc -O3 -arch=native -std=c++17 gemm_naive.cu -lcublas -o gemm_naive
// 运行：./gemm_naive > results/gemm_naive.csv
// ---------------------------------------------------------------------------
#include "gemm_bench.h"

__global__ void gemm_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

int main() {
    const DeviceInfo dev = query_device();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 强制严格 FP32：否则 cuBLAS 可能走允许降精度的 Tensor Core 路径，
    // 拿普通 CUDA Core 的 FP32 Kernel 去比，数字真实但结论不公平。
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::printf("# task1 gemm_naive benchmark\n");
    std::printf("# device   : %s (sm_%d%d)\n", dev.name.c_str(), dev.cc_major, dev.cc_minor);
    std::printf("# baseline : cuBLAS SGEMM, CUBLAS_PEDANTIC_MATH, FP32\n");
    std::printf("# measurement: warmup=%d, iters=%d, CUDA Event, kernel-only\n",
                kWarmupIters, kMeasureIters);
    std::printf("# correctness: max|err| < 1e-2 (vs cuBLAS)\n");
    print_gemm_csv_header();
    std::fflush(stdout);

    const dim3 block(16, 16);

    for (const GemmShape& shape : standard_gemm_shapes()) {
        const std::size_t a_bytes = static_cast<std::size_t>(shape.M) * shape.K * sizeof(float);
        const std::size_t b_bytes = static_cast<std::size_t>(shape.K) * shape.N * sizeof(float);
        const std::size_t c_bytes = static_cast<std::size_t>(shape.M) * shape.N * sizeof(float);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, a_bytes));
        CUDA_CHECK(cudaMalloc(&dB, b_bytes));
        CUDA_CHECK(cudaMalloc(&dC, c_bytes));
        init_gemm_inputs(dA, dB, shape.M, shape.N, shape.K);

        const dim3 grid((shape.N + block.x - 1) / block.x, (shape.M + block.y - 1) / block.y);
        auto launch = [&] {
            gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, shape.M, shape.N, shape.K);
        };

        bench_gemm_case(handle, "naive", shape, dA, dB, dC, launch);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
