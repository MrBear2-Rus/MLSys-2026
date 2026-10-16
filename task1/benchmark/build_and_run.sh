#!/usr/bin/env bash
# task1/benchmark/build_and_run.sh
# ---------------------------------------------------------------------------
# 一键编译 + 运行 Task 1 的全部 CUDA 基准，并把原始数据写进 ../results/
#
# 用法：
#   cd task1
#   bash benchmark/build_and_run.sh
#
# 依赖：nvcc（CUDA Toolkit）、有 NVIDIA GPU 的机器
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$(dirname "${SCRIPT_DIR}")"
CUDA_DIR="${TASK_DIR}/cuda"
RESULTS_DIR="${TASK_DIR}/results"
BUILD_DIR="${TASK_DIR}/build"

mkdir -p "${RESULTS_DIR}" "${BUILD_DIR}"

# ---------------------------------------------------------------------------
# 0. 环境记录（报告要求：GPU / Driver / CUDA / 编译参数）
# ---------------------------------------------------------------------------
ENV_FILE="${RESULTS_DIR}/environment.txt"
{
  echo "# Task 1 实验环境（由 build_and_run.sh 自动记录）"
  echo "## date";            date
  echo "## uname";           uname -a
  echo "## nvidia-smi";      nvidia-smi || echo "nvidia-smi: NOT FOUND"
  echo "## nvcc --version";  nvcc --version
  echo "## gcc --version";   gcc --version | head -1
  echo "## build flags";     echo "-O3 -arch=native -std=c++17 ; linked with -lcublas"
  echo "## measurement";     echo "warmup=10, iters=50, CUDA Event, kernel-only"
  echo "## seed";            echo "2026, FP32, Uniform[-1, 1]"
} > "${ENV_FILE}" 2>&1
echo "[env] wrote ${ENV_FILE}"

# ---------------------------------------------------------------------------
# 1. 编译（-arch=native 让 nvcc 直接面向本机算力；旧版本 nvcc 用 sm_70 兜底）
# ---------------------------------------------------------------------------
ARCH_FLAG="-arch=native"
if ! nvcc --help 2>&1 | grep -q -- "-arch=native"; then
  echo "[warn] nvcc 不支持 -arch=native，回退到 -arch=sm_70"
  ARCH_FLAG="-arch=sm_70"
fi

# cuBLAS 的库路径：多数环境（含 Colab）直接 -lcublas 即可，这里做一次兜底
CUDA_LIB_FLAG=""
for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
  if [ -d "${d}" ]; then CUDA_LIB_FLAG="-L${d}"; break; fi
done

build_one() {
  local src="$1" out="$2"
  echo "[build] ${src} -> ${out}"
  nvcc -O3 ${ARCH_FLAG} -std=c++17 -lineinfo ${CUDA_LIB_FLAG} \
       "${CUDA_DIR}/${src}" -lcublas -o "${BUILD_DIR}/${out}"
}

build_one vector_add.cu    vector_add
build_one gemm_naive.cu    gemm_naive
build_one gemm_shared.cu   gemm_shared
build_one gemm_register.cu gemm_register   # 完成 1.5 时才有；未完成可注释掉

# ---------------------------------------------------------------------------
# 2. 运行并保存原始输出（stdout = CSV，stderr 里也会出现正确性失败提示）
# ---------------------------------------------------------------------------
run_one() {
  local exe="$1" csv="$2"
  echo "[run ] ${exe} -> ${csv}"
  "${BUILD_DIR}/${exe}" > "${RESULTS_DIR}/${csv}" 2> "${RESULTS_DIR}/${csv%.csv}.stderr.txt"
}

run_one vector_add    vector_add.csv
run_one gemm_naive    gemm_naive.csv
run_one gemm_shared   gemm_shared.csv
run_one gemm_register gemm_register.csv

echo
echo "[done] 原始数据已写入 ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}"
echo
echo "提示：CSV 里以 '#' 开头的是环境与口径说明行，用 pandas 读取时可用 comment='#'。"
