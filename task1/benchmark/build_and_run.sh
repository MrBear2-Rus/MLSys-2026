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
# 1. 先确保 nvcc 在 PATH 上
#    租来的镜像常常是「conda 环境自带 CUDA runtime，但 toolkit 在 /usr/local/cuda」，
#    两者是两套东西：conda 的 cudatoolkit 不含 nvcc。
# ---------------------------------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [ -x "${d}/nvcc" ]; then export PATH="${d}:${PATH}"; echo "[path] 加入 ${d}"; break; fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] 找不到 nvcc，请确认镜像里装了 CUDA Toolkit（conda 里的 cudatoolkit 不含 nvcc）。"
  echo "        可以试：apt-get install -y cuda-toolkit-12-1  或  改用带 CUDA 的镜像。"
  exit 1
fi
echo "[env] nvcc: $(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------
# 2. 架构选择：优先从 nvidia-smi 读真实 compute capability（租的卡最稳）
#    -> -arch=native -> sm_70 兜底
# ---------------------------------------------------------------------------
ARCH_FLAG=""
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
case "${CC_RAW}" in
  ''|*[!0-9]*)
    if nvcc --help 2>&1 | grep -q -- "-arch=native"; then
      ARCH_FLAG="-arch=native"
      echo "[arch] 使用 -arch=native"
    else
      ARCH_FLAG="-arch=sm_70"
      echo "[warn] 探测不到算力且 nvcc 不支持 -arch=native，回退 sm_70"
    fi
    ;;
  *)
    ARCH_FLAG="-arch=sm_${CC_RAW}"
    echo "[arch] nvidia-smi 报告 compute capability ${CC_RAW} -> ${ARCH_FLAG}"
    ;;
esac

# cuBLAS 的库路径：多数环境直接 -lcublas 即可。
# CUDA 13 起部分容器镜像不再提供 libcublas.so 这个软链，只留 libcublas.so.13，
# 所以这里先探测 -lcublas 能不能链接，不行就退化成「直接链接 .so 文件全路径」。
CUDA_LIB_FLAG=""
for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
  if [ -d "${d}" ]; then CUDA_LIB_FLAG="-L${d}"; break; fi
done

CUBLAS_LINK="-lcublas"
PROBE_SRC="$(mktemp /tmp/_cublas_probe_XXXX.cpp)"
echo 'int main(){return 0;}' > "${PROBE_SRC}"
if ! nvcc -O1 -x cu ${CUDA_LIB_FLAG} "${PROBE_SRC}" ${CUBLAS_LINK} -o /tmp/_cublas_probe 2>/dev/null; then
  CANDIDATE="$(ls /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so* 2>/dev/null | head -1)"
  if [ -n "${CANDIDATE}" ] && nvcc -O1 -x cu ${CUDA_LIB_FLAG} "${PROBE_SRC}" "${CANDIDATE}" -o /tmp/_cublas_probe 2>/dev/null; then
    CUBLAS_LINK="${CANDIDATE}"
    echo "[link] -lcublas 不可用，改用 ${CANDIDATE}"
  else
    echo "[link] 无法链接 cuBLAS，编译大概率会失败；可换用 CUDA 12.x 的镜像重试。"
  fi
else
  echo "[link] cuBLAS 链接探测通过（${CUBLAS_LINK}）"
fi
rm -f "${PROBE_SRC}" /tmp/_cublas_probe

build_one() {
  local src="$1" out="$2"
  echo "[build] ${src} -> ${out}"
  nvcc -O3 ${ARCH_FLAG} -std=c++17 -lineinfo ${CUDA_LIB_FLAG} \
       "${CUDA_DIR}/${src}" ${CUBLAS_LINK} -o "${BUILD_DIR}/${out}"
}

build_one vector_add.cu    vector_add
build_one gemm_naive.cu    gemm_naive
build_one gemm_shared.cu   gemm_shared
build_one gemm_register.cu gemm_register

build_one vector_add.cu    vector_add
build_one gemm_naive.cu    gemm_naive
build_one gemm_shared.cu   gemm_shared
build_one gemm_register.cu gemm_register   # 完成 1.5 时才有；未完成可注释掉

# ---------------------------------------------------------------------------
# 3. 运行并保存原始输出（stdout = CSV，stderr 里也会出现正确性失败提示）
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
