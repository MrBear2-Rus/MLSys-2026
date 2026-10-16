#!/usr/bin/env bash
# MLSys-2026/run_all.sh
# ---------------------------------------------------------------------------
# 在「租来的 NVIDIA GPU 机器」上一条命令跑完全部实验。
#
# 用法（仓库根目录，也就是本文件所在目录）：
#     bash run_all.sh
#
# 它做四件事：
#   1. 打印环境（GPU / 驱动 / nvcc / torch / triton），缺失的依赖自动补装
#   2. Task 1：编译 4 个 .cu，跑 Vector Add + 3 种 GEMM，画图
#   3. Task 2：跑 Triton Vector Add / GEMM / RMSNorm / 72 配置 Autotuning
#   4. 把 results 打包成 mlsys_results.zip，直接下载回本机即可
#
# 任何一个实验失败都不会中断后面的步骤（结果会如实记录失败信息）。
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

hr() { printf '\n%s\n' "============================================================"; }

# ---------------------------------------------------------------------------
# 0. 环境
# ---------------------------------------------------------------------------
hr; echo "[0/5] 环境确认"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true
else
  echo "[ERROR] 没有 nvidia-smi：这台机器上没有可用的 NVIDIA 驱动，无法继续。"
  exit 1
fi

PY=python3
command -v ${PY} >/dev/null 2>&1 || PY=python

${PY} - <<'PYEOF' || true
import sys
print("python :", sys.version.split()[0])
try:
    import torch
    print("torch  :", torch.__version__, "| cuda:", torch.version.cuda,
          "| available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device :", torch.cuda.get_device_name(0))
        print("cap    : sm_%d%d" % torch.cuda.get_device_capability(0))
    else:
        print("device : torch 看不到 GPU（可能是装了 CPU 版 torch）")
except Exception as exc:
    print("torch  : NOT INSTALLED ->", exc)
try:
    import triton
    print("triton :", triton.__version__)
except Exception as exc:
    print("triton : NOT INSTALLED ->", exc)
try:
    import pandas, matplotlib
    print("pandas :", pandas.__version__, "| matplotlib:", matplotlib.__version__)
except Exception as exc:
    print("plot deps: MISSING ->", exc)
PYEOF

# ---------------------------------------------------------------------------
# 1. 依赖补装（只装缺的，不覆盖已有环境）
# ---------------------------------------------------------------------------
hr; echo "[1/5] 检查并按需安装 Python 依赖"
MISSING=""
${PY} -c "import torch" 2>/dev/null || MISSING="${MISSING} torch"
${PY} -c "import triton" 2>/dev/null || MISSING="${MISSING} triton"
${PY} -c "import pandas" 2>/dev/null || MISSING="${MISSING} pandas"
${PY} -c "import matplotlib" 2>/dev/null || MISSING="${MISSING} matplotlib"

if [ -n "${MISSING}" ]; then
  echo "[pip] 缺失：${MISSING}"
  ${PY} -m pip install -q ${MISSING} || ${PY} -m pip install -q --index-url https://pypi.tuna.tsinghua.edu.cn/simple ${MISSING}
else
  echo "[pip] 依赖齐全，跳过安装"
fi

# torch 装了 CPU 版的话，任务跑不出 GPU 结果，这里提醒一下
${PY} -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || echo "[warn] torch 看不到 CUDA —— Task 2 会失败，请安装 CUDA 版 torch（见 租卡指南.md）"

# ---------------------------------------------------------------------------
# 2. Task 1
# ---------------------------------------------------------------------------
hr; echo "[2/5] Task 1：CUDA（Vector Add / Naive GEMM / Shared Tiling / Register Tiling）"
if bash task1/benchmark/build_and_run.sh; then
  echo "[ok] Task 1 完成"
else
  echo "[warn] Task 1 有失败项，请查看 task1/results/*.stderr.txt"
fi

# ---------------------------------------------------------------------------
# 3. Task 1 画图
# ---------------------------------------------------------------------------
hr; echo "[3/5] Task 1：画图"
${PY} task1/benchmark/plot_results.py || echo "[warn] 画图失败（可能是某个 CSV 为空）"

# ---------------------------------------------------------------------------
# 4. Task 2
# ---------------------------------------------------------------------------
hr; echo "[4/5] Task 2：Triton（Vector Add / GEMM / RMSNorm / Autotuning）"
if (cd task2 && ${PY} triton/bench_all.py); then
  echo "[ok] Task 2 完成"
else
  echo "[warn] Task 2 有失败项，上方日志为准"
fi

# ---------------------------------------------------------------------------
# 5. 打包
# ---------------------------------------------------------------------------
hr; echo "[5/5] 打包结果"
${PY} - <<'PYEOF'
import os, zipfile, glob
out = "mlsys_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for base in ("task1/results", "task2/results"):
        for p in glob.glob(base + "/**", recursive=True):
            if os.path.isfile(p):
                zf.write(p, p)
    for extra in ("task0/env/env_raw.txt",):
        if os.path.isfile(extra):
            zf.write(extra, extra)
print("打包完成 ->", out)
with zipfile.ZipFile(out) as zf:
    names = zf.namelist()
    print("包含", len(names), "个文件：")
    for n in names:
        print("  ", n)
PYEOF

hr
echo "全部结束。下一步："
echo "  1. 下载 mlsys_results.zip（AutoDL 用 JupyterLab 左侧文件区右键下载）"
echo "  2. 解压后把 environment.txt / *.csv / figures/*.png 发给我，我帮你补齐两份报告里的 [待填]"
echo "  3. 顺手把 results 提交回仓库：git add -A && git commit -m '补齐实测数据' && git push"
