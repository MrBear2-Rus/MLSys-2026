// task1/cuda/gemm_register.cu
// ---------------------------------------------------------------------------
// Task 1 - 1.5 进阶：Register Tiling
//
// 在 Shared Memory Tiling 的基础上，让"一个 thread 算 4x4 = 16 个 C 元素"，
// 把 partial sum 尽可能留在 Register 里。
//
// Block 组织：
//   TILE = 32（Block 负责 32x32 的 C 分块）
//   blockDim = (8, 8) = 64 threads，每个 thread 负责 4x4 的输出子块
//   每个 k 步：
//     从 shared memory 读 4 个 A 值 + 4 个 B 值（共 8 次 shared load）
//     做 16 次 FMA
//   -> 算术强度比"每 thread 1 个 C 元素"提升约 2 倍（8 load : 16 FMA）
//
// 关于 Bank Conflict：
//   若声明为 As[TILE][TILE]，行距 32 个字 = 32 banks，同一列的不同行会撞同一个 bank。
//   这里用 [TILE][TILE+1] 做 padding，行距变成 33，错开 bank，消除该冲突。
//
// 编译：nvcc -O3 -arch=native -std=c++17 gemm_register.cu -lcublas -o gemm_register
// 运行：./gemm_register > results/gemm_register.csv
// ---------------------------------------------------------------------------
#include "gemm_bench.h"

namespace {
constexpr int TILE = 32;       // Block 负责的 C 分块边长
constexpr int TM   = 4;        // 每个 thread 负责的行数
constexpr int TN   = 4;        // 每个 thread 负责的列数
constexpr int THREADS_X = TILE / TN;  // 8
constexpr int THREADS_Y = TILE / TM;  // 8
}  // namespace

__global__ void gemm_register_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
    // padding 到 TILE+1，避免"同列不同行"落到同一个 bank
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;
    const int nthreads = THREADS_X * THREADS_Y;

    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;

    float acc[TM][TN];
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // ---- 协作搬运 32x32 的 A / B 分块（越界补 0）----
        for (int idx = tid; idx < TILE * TILE; idx += nthreads) {
            const int r = idx / TILE;
            const int c = idx % TILE;

            const int a_row = row0 + r;
            const int a_col = k0 + c;
            As[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

            const int b_row = k0 + r;
            const int b_col = col0 + c;
            Bs[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        // ---- 计算：8 次 shared load -> 16 次 FMA，accumulator 留在寄存器 ----
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float aReg[TM];
            float bReg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                aReg[i] = As[ty * TM + i][k];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                bReg[j] = Bs[k][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }
        __syncthreads();
    }

    // ---- 写回（越界不写）----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = row0 + ty * TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = col0 + tx * TN + j;
            if (col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

int main() {
    const DeviceInfo dev = query_device();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::printf("# task1 gemm_register benchmark (shared tiling + 4x4 register tiling)\n");
    std::printf("# device   : %s (sm_%d%d)\n", dev.name.c_str(), dev.cc_major, dev.cc_minor);
    std::printf("# blockTile=%d, threadTile=%dx%d, blockDim=(%d,%d)\n", TILE, TM, TN,
                THREADS_X, THREADS_Y);
    std::printf("# baseline : cuBLAS SGEMM, CUBLAS_PEDANTIC_MATH, FP32\n");
    std::printf("# measurement: warmup=%d, iters=%d, CUDA Event, kernel-only\n",
                kWarmupIters, kMeasureIters);
    print_gemm_csv_header();
    std::fflush(stdout);

    const dim3 block(THREADS_X, THREADS_Y);

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
            gemm_register_kernel<<<grid, block>>>(dA, dB, dC, shape.M, shape.N, shape.K);
        };

        bench_gemm_case(handle, "register-4x4", shape, dA, dB, dC, launch);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
