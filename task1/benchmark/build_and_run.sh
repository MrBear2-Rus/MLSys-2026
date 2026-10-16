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
# 2. 架构选择：**逐个探测**哪组 -arch 能编译通过
#    为什么要探测而不是直接用 -arch=native：
#      - 50 系（RTX 5070 等）= Blackwell sm_120，需要 CUDA Toolkit 12.8+ 才认识；
#        旧 nvcc 会直接报 "Unsupported gpu architecture 'compute_120'"。
#      - 所以候选顺序是：nvidia-smi 读到的真实算力 -> 它对应的 PTX -> native -> 保守兜底。
# ---------------------------------------------------------------------------
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
CANDIDATES=""
case "${CC_RAW}" in
  ''|*[!0-9]*) ;;
  *)
    echo "[arch] nvidia-smi 报告 compute capability ${CC_RAW}"
    # 1) 精确匹配：生成该架构的原生 SASS，最快
    CANDIDATES="sm_${CC_RAW} compute_${CC_RAW}"
    ;;
esac
# 2) 兜底：一律用 **PTX**（compute_XX）而不是 sm_XX。
#    因为 cubin/SASS 不跨大版本兼容，sm_90 的二进制在 sm_120（5070）上会直接
#    报 "no kernel image is available"；而 PTX 是前向兼容的，驱动会在首次运行时 JIT。
CANDIDATES="${CANDIDATES} native compute_120 compute_90 compute_86 compute_75"

PROBE_SRC="$(mktemp /tmp/_arch_probe_XXXX.cu)"
printf '__global__ void _probe(void){}\nint main(void){return 0;}\n' > "${PROBE_SRC}"

ARCH_FLAG=""
USED_PTX_FALLBACK=0
for cand in ${CANDIDATES}; do
  case "${cand}" in
    compute_*)
      NUM="${cand#compute_}"
      FLAG="-gencode=arch=compute_${NUM},code=compute_${NUM}"
      ;;
    *)
      FLAG="-arch=${cand}"
      ;;
  esac
  if nvcc -O1 ${FLAG} "${PROBE_SRC}" -o /tmp/_arch_probe.bin >/dev/null 2>&1; then
    ARCH_FLAG="${FLAG}"
    echo "[arch] 采用 ${FLAG}"
    break
  fi
done
rm -f "${PROBE_SRC}" /tmp/_arch_probe.bin

if [ -z "${ARCH_FLAG}" ]; then
  echo "[ERROR] 没有任何 -arch 候选能编译通过。"
  echo "        50 系显卡请把 CUDA Toolkit 升级到 12.8 或更高（旧版 nvcc 不认识 sm_120）。"
  echo "        当前：$(nvcc --version | tail -1)"
  exit 1
fi

# 如果没用上精确架构，提醒一下：PTX 回退会导致第一次 kernel 启动变慢（驱动 JIT），
# 而且少数情况下运行时会报 no kernel image —— 那时唯一解是升级 CUDA Toolkit。
case "${ARCH_FLAG}" in
  *"compute_${CC_RAW}"*|*"sm_${CC_RAW}"*|*native*) ;;
  *) USED_PTX_FALLBACK=1 ;;
esac
if [ "${USED_PTX_FALLBACK}" -eq 1 ]; then
  echo "[warn] 没有用上本卡原生的 sm_${CC_RAW}（nvcc 版本偏旧）。"
  echo "       已改用 PTX 前向兼容模式：能跑，但首次 kernel 启动会慢一点。"
  echo "       若运行时报 'no kernel image is available'，请升级 CUDA Toolkit 后重试。"
fi

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
