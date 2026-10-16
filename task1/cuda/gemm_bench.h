// task1/cuda/gemm_bench.h
// ---------------------------------------------------------------------------
// GEMM 变体共用的 benchmark 骨架（Naive / Shared Tiling / Register Tiling 复用）
//
// 统一口径：
//   - 正确性：与"严格 FP32 的 cuBLAS"结果比较，max|err| < 1e-2；
//   - 计时  ：CUDA Event，warmup 10 + measure 50，报告 median / P95；
//   - 只测 Kernel，不含 cudaMalloc / cudaMemcpy；
//   - cuBLAS 基线强制 CUBLAS_PEDANTIC_MATH，避免拿 FP32 核去比 Tensor Core 降精度路径。
//
// 行主序如何映射到列主序 cuBLAS：
//   行主序 C(M,N) = A(M,K) * B(K,N)  等价于列主序 C^T(N,M) = B^T(N,K) * A^T(K,M)
//   所以 cublasSgemm(op=N, op=N, m=N, n=M, k=K, A=B, lda=N, B=A, ldb=K, C=C, ldc=N)
// ---------------------------------------------------------------------------
#pragma once

#include "cuda_utils.h"

#include <functional>
#include <string>
#include <vector>

struct GemmShape {
    int         M;
    int         N;
    int         K;
    std::string tag;
};

// 题面给出的标准实验矩阵
inline std::vector<GemmShape> standard_gemm_shapes() {
    return {
        {2048, 2048, 2048, "square"},        // 方阵
        {4096, 64, 4096, "tall-skinny"},     // 高瘦形
        {1, 4096, 4096, "small-M-1"},        // 小 M
        {32, 4096, 4096, "small-M-32"},
        {256, 4096, 4096, "small-M-256"},
        {127, 131, 129, "irregular"},        // 不规则
    };
}

// 行主序 -> 列主序 cuBLAS 包装（严格 FP32）
inline void cublas_sgemm_fp32(cublasHandle_t handle, const float* dA, const float* dB,
                              float* dC, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, /*m=*/N, /*n=*/M, /*k=*/K,
                             &alpha, dB, /*lda=*/N, dA, /*ldb=*/K, &beta, dC, /*ldc=*/N));
}

// ---------------------------------------------------------------------------
// 跑一个 (variant, shape) 组合并打印一行 CSV
// ---------------------------------------------------------------------------
template <typename LaunchFn>
void bench_gemm_case(cublasHandle_t handle, const std::string& variant,
                     const GemmShape& shape, float* dA, float* dB, float* dC,
                     LaunchFn&& launch) {
    const std::size_t c_elems = static_cast<std::size_t>(shape.M) * shape.N;
    const std::size_t c_bytes = c_elems * sizeof(float);
    const double flops = 2.0 * shape.M * shape.N * shape.K;

    std::vector<float> h_custom(c_elems), h_ref(c_elems);

    // ---- 1) 自写 Kernel：正确性 ----
    CUDA_CHECK(cudaMemset(dC, 0, c_bytes));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_custom.data(), dC, c_bytes, cudaMemcpyDeviceToHost));

    // ---- 2) cuBLAS 严格 FP32 参考 ----
    CUDA_CHECK(cudaMemset(dC, 0, c_bytes));
    cublas_sgemm_fp32(handle, dA, dB, dC, shape.M, shape.N, shape.K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ref.data(), dC, c_bytes, cudaMemcpyDeviceToHost));

    const double err = max_abs_error(h_custom, h_ref);

    // ---- 3) 自写 Kernel：性能 ----
    const std::vector<double> custom_samples = benchmark_kernel(launch);
    const double custom_med = median_of(custom_samples);
    const double custom_p95 = p95_of(custom_samples);
    const double custom_tflops = flops / (custom_med * 1e-3) / 1e12;

    // ---- 4) cuBLAS：性能（同一套口径）----
    auto cublas_launch = [&] {
        cublas_sgemm_fp32(handle, dA, dB, dC, shape.M, shape.N, shape.K);
    };
    const std::vector<double> cublas_samples = benchmark_kernel(cublas_launch);
    const double cublas_med = median_of(cublas_samples);
    const double cublas_tflops = flops / (cublas_med * 1e-3) / 1e12;
    const double pct = 100.0 * custom_tflops / cublas_tflops;

    std::printf("%s,%s,%d,%d,%d,%.6e,%.6f,%.6f,%.4f,%.6f,%.4f,%.2f%s\n", variant.c_str(),
                shape.tag.c_str(), shape.M, shape.N, shape.K, err, custom_med, custom_p95,
                custom_tflops, cublas_med, cublas_tflops, pct,
                (err < 1e-2) ? "" : "   <-- FAILED CORRECTNESS");
    std::fflush(stdout);
}

inline void print_gemm_csv_header() {
    std::printf(
        "variant,shape,M,N,K,max_abs_err,median_ms,p95_ms,TFLOPS,"
        "cublas_median_ms,cublas_TFLOPS,pct_of_cublas\n");
}
