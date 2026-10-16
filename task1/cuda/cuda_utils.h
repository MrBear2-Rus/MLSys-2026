// task1/cuda/cuda_utils.h
// ---------------------------------------------------------------------------
// Task 1 公共工具：错误检查、输入生成、CUDA Event 计时、统计量、CSV 输出。
//
// 约定（与题面「开始实验前，先把比较口径对齐」保持一致）：
//   1. 输入：FP32、Uniform[-1, 1]、seed = 2026；
//   2. 计时：CUDA Event，warmup 10 次，正式测量 50 次，报告 median 与 P95；
//   3. 默认只统计 Kernel 执行，不含 cudaMalloc / cudaMemcpy / 初始化；
//   4. 先过正确性，再看性能；逐次原始数据全部保留。
// ---------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 1. 错误检查
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s -> %s\n", __FILE__,    \
                         __LINE__, #call, cudaGetErrorString(_err));            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _st = (call);                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "[CUBLAS ERROR] %s:%d  %s -> status %d\n",     \
                         __FILE__, __LINE__, #call, static_cast<int>(_st));     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// 2. 实验超参（题面约定，集中放这里，方便报告里引用）
// ---------------------------------------------------------------------------
constexpr int    kWarmupIters = 10;     // 预热次数
constexpr int    kMeasureIters = 50;    // 正式测量次数
constexpr int    kSeed        = 2026;   // 标准 seed
constexpr float  kRangeLo     = -1.0f;  // Uniform 下界
constexpr float  kRangeHi     = 1.0f;   // Uniform 上界

// ---------------------------------------------------------------------------
// 3. 输入生成：FP32 + Uniform[-1, 1] + seed 2026
// ---------------------------------------------------------------------------
inline void fill_uniform(std::vector<float>& host, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(kRangeLo, kRangeHi);
    for (float& v : host) {
        v = dist(rng);
    }
}

// 生成 1D 向量并拷到设备（Vector Add 用）
inline void init_vector_inputs(float* d_A, float* d_B, std::size_t n) {
    std::mt19937 rng(kSeed);
    std::vector<float> h_A(n), h_B(n);
    fill_uniform(h_A, rng);
    fill_uniform(h_B, rng);
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// 生成矩阵输入并拷到设备（GEMM 用）
inline void init_gemm_inputs(float* d_A, float* d_B, std::size_t m, std::size_t n,
                             std::size_t k) {
    std::mt19937 rng(kSeed);
    std::vector<float> h_A(m * k), h_B(k * n);
    fill_uniform(h_A, rng);
    fill_uniform(h_B, rng);
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// 4. CUDA Event 计时器
//    只包住 kernel 调用区间，不含内存申请与拷贝。
// ---------------------------------------------------------------------------
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop()  { CUDA_CHECK(cudaEventRecord(stop_)); }
    // 返回上一次 start -> stop 之间的毫秒数
    double elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms);
    }

private:
    cudaEvent_t start_{}, stop_{};
};

// ---------------------------------------------------------------------------
// 5. 统计量
// ---------------------------------------------------------------------------
inline double percentile(std::vector<double> samples, double p) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const double idx = p * static_cast<double>(samples.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) return samples[lo];
    const double frac = idx - static_cast<double>(lo);
    return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}

inline double median_of(std::vector<double> samples) { return percentile(samples, 0.50); }
inline double p95_of(std::vector<double> samples)    { return percentile(samples, 0.95); }

// ---------------------------------------------------------------------------
// 6. 设备信息 / 理论带宽
// ---------------------------------------------------------------------------
struct DeviceInfo {
    std::string name;
    int         cc_major = 0;
    int         cc_minor = 0;
    int         sm_count = 0;
    int         max_threads_per_block = 0;
    int         max_threads_per_sm = 0;
    int         regs_per_sm = 0;
    int         shared_mem_per_block = 0;
    double      peak_bw_gbps = 0.0;  // 理论显存带宽
};

inline DeviceInfo query_device() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    DeviceInfo info;
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    info.regs_per_sm = prop.regsPerMultiprocessor;
    info.shared_mem_per_block = static_cast<int>(prop.sharedMemPerBlock);
    // 理论带宽 = 显存频率(kHz) * 位宽(bit) / 8 * 2(DDR) -> GB/s
    info.peak_bw_gbps = static_cast<double>(prop.memoryClockRate) * 1000.0 *
                        static_cast<double>(prop.memoryBusWidth) / 8.0 * 2.0 / 1e9;
    return info;
}

// ---------------------------------------------------------------------------
// 7. Benchmark 主循环（统一口径：warmup 10 + measure 50，返回逐次原始耗时）
//    launch_fn 是"一次 kernel 调用"，由各 .cu 用 lambda 传入。
// ---------------------------------------------------------------------------
template <typename LaunchFn>
std::vector<double> benchmark_kernel(LaunchFn&& launch_fn, int warmup = kWarmupIters,
                                     int measure = kMeasureIters) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(measure));

    CudaTimer timer;
    for (int i = 0; i < warmup; ++i) {
        launch_fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < measure; ++i) {
        timer.start();
        launch_fn();
        timer.stop();
        samples.push_back(timer.elapsed_ms());
    }
    return samples;
}

// ---------------------------------------------------------------------------
// 8. 正确性检查：最大绝对误差
// ---------------------------------------------------------------------------
inline double max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return worst;
}

// ---------------------------------------------------------------------------
// 9. CSV 输出小工具
// ---------------------------------------------------------------------------
inline void csv_header(const char* path) {
    // 每个 .cu 自己打印表头，方便直接 cat 进 results/*.csv
    std::printf("%s\n", path);
}
