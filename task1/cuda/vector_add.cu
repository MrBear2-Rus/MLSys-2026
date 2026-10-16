// task1/cuda/vector_add.cu
// ---------------------------------------------------------------------------
// Task 1 - 1.2 Hello CUDA：Vector Add
//
//   C_i = A_i + B_i
//
// 目标：把"数据从 CPU 到 GPU、Thread 找到自己负责的元素、结果回到 CPU"
//       这条完整链路走通，再谈性能。
//
// 编译：
//   nvcc -O3 -arch=native -std=c++17 vector_add.cu -o vector_add
// 运行：
//   ./vector_add > results/vector_add.csv
// ---------------------------------------------------------------------------
#include "cuda_utils.h"

#include <cstdio>

// ---------------------------------------------------------------------------
// Kernel：i = blockIdx.x * blockDim.x + threadIdx.x
//   - 一维 grid，一个 thread 负责一个元素；
//   - 做边界判断，保证最后一个 block 不会越界；
//   - 访问连续，天然满足 memory coalescing。
// ---------------------------------------------------------------------------
__global__ void vector_add_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// CPU 参考实现：正确性检验的基准
static void vector_add_cpu(const std::vector<float>& a, const std::vector<float>& b,
                           std::vector<float>& c) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const DeviceInfo dev = query_device();

    // ---- 环境信息：报告里要求的 GPU / Driver / CUDA / 编译参数 ----
    std::printf("# task1 vector_add benchmark\n");
    std::printf("# device            : %s (sm_%d%d, %d SMs)\n", dev.name.c_str(),
                dev.cc_major, dev.cc_minor, dev.sm_count);
    std::printf("# max threads/block : %d\n", dev.max_threads_per_block);
    std::printf("# shared mem/block  : %d bytes\n", dev.shared_mem_per_block);
    std::printf("# peak bandwidth    : %.1f GB/s\n", dev.peak_bw_gbps);
    std::printf("# measurement       : warmup=%d, iters=%d, timer=CUDA Event, "
                "kernel-only\n", kWarmupIters, kMeasureIters);
    std::printf("# build flags       : -O3 -arch=native -std=c++17\n");
    std::printf("# correctness       : max|err| < 1e-5 (vs CPU reference)\n");
    std::printf("N,block_size,max_abs_err,median_ms,p95_ms,effective_GBps,peak_bw_util_pct\n");
    std::fflush(stdout);

    // 标准 Shape；显存不足时可改为 {2^20, 2^22}（题面允许）
    const std::vector<int> sizes = {1 << 20, 1 << 22, 1 << 24};
    const std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};

    for (int n : sizes) {
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        // 初始化和 H2D 拷贝都放在计时循环之外（否则测到的就不只是 Kernel）
        init_vector_inputs(d_A, d_B, static_cast<std::size_t>(n));

        for (int bs : block_sizes) {
            // 超过设备上限的 Block Size 直接跳过，并在报告里记录设备限制
            if (bs > dev.max_threads_per_block) {
                std::printf("# SKIP N=%d block_size=%d (> max threads/block %d)\n", n,
                            bs, dev.max_threads_per_block);
                continue;
            }

            const int grid = (n + bs - 1) / bs;
            auto launch = [&] { vector_add_kernel<<<grid, bs>>>(d_A, d_B, d_C, n); };

            // ---- 1) 先正确性 ----
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> h_C(n), h_A(n), h_B(n), h_ref(n);
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, bytes, cudaMemcpyDeviceToHost));
            vector_add_cpu(h_A, h_B, h_ref);

            const double err = max_abs_error(h_C, h_ref);
            const bool ok = err < 1e-5;

            // ---- 2) 再性能 ----
            const std::vector<double> samples = benchmark_kernel(launch);
            const double med = median_of(samples);
            const double p95 = p95_of(samples);
            // 有效带宽：读 A(4N) + 读 B(4N) + 写 C(4N) = 12N Bytes
            const double effective_gbps =
                (12.0 * static_cast<double>(n)) / (med * 1e-3) / 1e9;

            std::printf("%d,%d,%.6e,%.6f,%.6f,%.2f,%.2f%s\n", n, bs, err, med, p95,
                        effective_gbps, 100.0 * effective_gbps / dev.peak_bw_gbps,
                        ok ? "" : "   <-- FAILED CORRECTNESS");
            std::fflush(stdout);

            if (!ok) {
                std::fprintf(stderr, "[CORRECTNESS FAILED] N=%d bs=%d max_err=%.6e\n", n,
                             bs, err);
            }
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    return 0;
}
