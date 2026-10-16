#!/usr/bin/env bash
# MLSys-2026/run_all.sh
# ---------------------------------------------------------------------------
# 在「任意一台有 NVIDIA GPU 的 Linux / WSL2 机器」上一条命令跑完全部实验。
#
# 用法（仓库根目录）：  bash run_all.sh
#
# 会自动处理的环境差异：
#   - Python 是 PEP 668 externally-managed（Ubuntu 23.04+/24.04 默认）-> 自动建 .venv
#   - torch 未装 / 是 CPU 版 -> 按 GPU 算力选对 CUDA 轮子（Blackwell 用 cu128）
#   - nvcc 不在 PATH -> 先去 /usr/local/cuda* 找；找不到就给出精确安装命令
#   - 缺 pandas/matplotlib/triton -> 自动补装（失败自动换清华源）
#
# 任何一步失败都不会中断后面的步骤：比如没装 CUDA Toolkit 时 Task 1 会跳过，
# 但 Task 2（只需要 torch+triton）照常跑。
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

hr() { printf '\n%s\n' "============================================================"; }

# ---------------------------------------------------------------------------
# 0. 环境确认
# ---------------------------------------------------------------------------
hr; echo "[0/6] 环境确认"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
             --format=csv,noheader || true
else
  echo "[ERROR] 没有 nvidia-smi：这台机器没有可用的 NVIDIA 驱动。"
  echo "        WSL 下请确认：宿主 Windows 已装 NVIDIA 驱动，且本脚本在 WSL 内部执行。"
  exit 1
fi
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
echo "[gpu] compute capability = ${CC_RAW:-未知}"

# Python 选择：仓库内 .venv 优先，其次系统 python3
BASE_PY="$(command -v python3 || command -v python)"
PY="${ROOT}/.venv/bin/python"
[ -x "${PY}" ] || PY="${BASE_PY}"
echo "[py]  使用 ${PY} ($("${PY}" --version 2>&1))"

"${PY}" - <<'PYEOF' || true
def show(name):
    try:
        mod = __import__(name)
        print(f"  {name:12s}: {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:
        print(f"  {name:12s}: MISSING -> {exc}")
show("torch")
show("triton")
show("pandas")
show("matplotlib")
PYEOF

# nvcc 提前检查（不要等到 Task 1 编译时才发现，也把安装命令直接给出来）
NVCC_BIN="$(command -v nvcc || true)"
if [ -z "${NVCC_BIN}" ]; then
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [ -x "${d}/nvcc" ]; then NVCC_BIN="${d}/nvcc"; break; fi
  done
fi
if [ -n "${NVCC_BIN}" ]; then
  echo "[nvcc] ${NVCC_BIN}  ($("${NVCC_BIN}" --version 2>/dev/null | tail -1))"
else
  echo "[nvcc] 未找到 —— Task 1（CUDA）这次会跳过，但**不影响 Task 2 照常跑**。"
  echo "       装 CUDA Toolkit（WSL/Ubuntu，选 12.8+ 才认识 50 系 sm_120）："
  echo "         wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "         sudo dpkg -i cuda-keyring_1.1-1_all.deb"
  echo "         sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8"
  echo "       装完重新跑 bash run_all.sh 即可补上 Task 1。"
fi

# 把环境信息固化到 task0/env/env_raw.txt（供报告直接使用，避免手写错位 / 残留旧机环境）
mkdir -p task0/env
{
  echo "# 运行环境（由 run_all.sh 自动记录）"
  echo "## date"; date 2>/dev/null || true
  echo "## nvidia-smi"; nvidia-smi 2>/dev/null || echo "NOT FOUND"
  echo "## gpu"; nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
  echo "## nvcc"; nvcc --version 2>/dev/null || echo "NOT FOUND（未装 CUDA Toolkit）"
  echo "## torch"; "${PY}" -c "import torch,sys; print(torch.__version__,'| cuda runtime',torch.version.cuda,'| device',(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'))" 2>/dev/null || echo "NOT FOUND"
  echo "## triton"; "${PY}" -c "import triton; print(triton.__version__)" 2>/dev/null || echo "NOT FOUND"
} > task0/env/env_raw.txt 2>&1 || true
echo "[env] 已写入 task0/env/env_raw.txt"

# ---------------------------------------------------------------------------
# 1. 依赖安装（torch 按算力选轮子；PEP 668 自动改用 venv）
# ---------------------------------------------------------------------------
hr; echo "[1/6] 安装依赖（只装缺的）"

pip_install() {
  # 先官方源，失败自动换清华源
  "${PY}" -m pip install "$@" 2>&1 | tee -a install.log
  local st="${PIPESTATUS[0]}"
  if [ "${st}" -ne 0 ]; then
    echo "[pip] 官方源失败，换清华源重试 ..."
    "${PY}" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple "$@" 2>&1 | tee -a install.log
    st="${PIPESTATUS[0]}"
  fi
  return "${st}"
}

# 有任何缺失时才动 pip；若被 PEP 668 拦（externally-managed），就建仓库内 venv 再装
MISSING=""
"${PY}" -c "import torch"      >/dev/null 2>&1 || MISSING="${MISSING} torch"
"${PY}" -c "import triton"    >/dev/null 2>&1 || MISSING="${MISSING} triton"
"${PY}" -c "import pandas"     >/dev/null 2>&1 || MISSING="${MISSING} pandas"
"${PY}" -c "import matplotlib"  >/dev/null 2>&1 || MISSING="${MISSING} matplotlib"

if [ -n "${MISSING}" ]; then
  echo "[deps] 缺失：${MISSING}"

  # 先试一次；如果是因为 externally-managed（PEP 668）失败，就建 venv
  if ! "${PY}" -m pip install --dry-run --quiet pip >/dev/null 2>&1; then
    echo "[venv] 系统 python 拒绝直接装包（PEP 668 externally-managed），"
    echo "       在仓库内创建 .venv（不碰系统 Python）..."
    if "${BASE_PY}" -m venv "${ROOT}/.venv" >/dev/null 2>&1; then
      PY="${ROOT}/.venv/bin/python"
      "${PY}" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
      echo "[venv] 已创建并切换到 ${PY}"
    else
      echo "[error] 创建 venv 失败 —— 通常是因为缺 python3-venv 包。"
      echo "        请执行：  sudo apt-get install -y python3-venv python3-full"
      echo "        然后重新运行 bash run_all.sh"
      exit 1
    fi
  fi

  # torch 要按算力选对 CUDA 轮子：Blackwell(sm_120, 50 系)必须 cu128 及以上
  if ! "${PY}" -c "import torch" >/dev/null 2>&1; then
    echo "[torch] 开始安装 CUDA 版 torch（轮子约 2~3 GB，耐心等待）..."
    TORCH_OK=0
    if [ "${CC_RAW:-0}" -ge 120 ] 2>/dev/null; then
      "${PY}" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee -a install.log \
        && TORCH_OK=1
    fi
    if [ "${TORCH_OK}" -eq 0 ]; then
      # 兜底：非 Blackwell 用 cu121；再不行用 PyPI 默认
      "${PY}" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tee -a install.log \
        && TORCH_OK=1
    fi
    if [ "${TORCH_OK}" -eq 0 ]; then
      "${PY}" -m pip install torch 2>&1 | tee -a install.log && TORCH_OK=1
    fi
    [ "${TORCH_OK}" -eq 1 ] || echo "[warn] torch 安装失败：Task 2 无法运行，把上面的报错发回来。"
  fi

  # torch 之外的其他依赖
  EXTRA=""
  "${PY}" -c "import triton"    >/dev/null 2>&1 || EXTRA="${EXTRA} triton"
  "${PY}" -c "import pandas"     >/dev/null 2>&1 || EXTRA="${EXTRA} pandas"
  "${PY}" -c "import matplotlib" >/dev/null 2>&1 || EXTRA="${EXTRA} matplotlib"
  if [ -n "${EXTRA}" ]; then
    pip_install ${EXTRA} || echo "[warn] 依赖 ${EXTRA} 安装失败，把上面的报错发回来。"
  fi
else
  echo "[deps] 依赖齐全，跳过安装"
fi

# torch 是否真的能看到 GPU
if ! "${PY}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[warn] torch 看不到 CUDA —— 可能装到了 CPU 版。"
  echo "       50 系（sm_120）请用：pip install torch --index-url https://download.pytorch.org/whl/cu128"
else
  echo "[torch] CUDA 可用：$("${PY}" -c "import torch; print(torch.cuda.get_device_name(0), '| torch cuda:', torch.version.cuda)")"
fi

# ---------------------------------------------------------------------------
# 1.5 冒烟测试：1 秒内确认 torch / triton 在这张卡上真的能跑
# ---------------------------------------------------------------------------
hr; echo "[1.5/6] 冒烟测试（torch 矩阵乘 + triton kernel）"
"${PY}" - <<'PYEOF' || echo "[warn] 冒烟测试未通过：Task 2 可能会失败，先看上面的具体报错"
import sys
try:
    import torch
except Exception as exc:
    print("  [fail] import torch:", exc); sys.exit(1)
if not torch.cuda.is_available():
    print("  [fail] torch.cuda.is_available() == False"); sys.exit(1)

a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
err = (a @ b - torch.matmul(a, b)).abs().max().item()
print(f"  [ok] torch GPU 矩阵乘正常 (max diff {err:.2e}), device={torch.cuda.get_device_name(0)}")

try:
    import triton
    import triton.language as tl
except Exception as exc:
    print("  [fail] import triton:", exc, "-> 试 pip install triton"); sys.exit(1)

@triton.jit
def _smoke(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

try:
    x = torch.ones(4096, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    _smoke[(4,)](x, y, 4096, BLOCK=1024, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    assert abs(y.mean().item() - 2.0) < 1e-5, y.mean().item()
    print(f"  [ok] triton {triton.__version__} kernel 运行正常")
except TypeError as exc:
    print("  [fail] triton 调用签名不兼容（triton 与 torch 版本不匹配）:", exc)
    print("         -> 解决：pip install -U triton  或按 torch 版本装对应 triton")
    sys.exit(1)
except Exception as exc:
    print("  [fail] triton kernel 执行失败:", exc)
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# 2. Task 1：CUDA（需要 nvcc；没有就跳过，不挡 Task 2）
# ---------------------------------------------------------------------------
hr; echo "[2/6] Task 1：CUDA（Vector Add / Naive GEMM / Shared Tiling / Register Tiling）"
if [ -n "${NVCC_BIN}" ] || command -v nvcc >/dev/null 2>&1; then
  if bash task1/benchmark/build_and_run.sh; then
    echo "[ok] Task 1 完成"
  else
    echo "[warn] Task 1 有失败项："
    echo "       - 看本目录 task1/build_and_run.log（若存在）"
    echo "       - 看各 task1/results/*.stderr.txt"
  fi
else
  echo "[skip] 没有 nvcc，跳过 Task 1（Task 2 不受影响）。"
  echo "       安装方法见上面 [nvcc] 段落；装完重跑 bash run_all.sh 补上。"
fi

# ---------------------------------------------------------------------------
# 3. Task 1 画图
# ---------------------------------------------------------------------------
hr; echo "[3/6] Task 1：画图"
"${PY}" task1/benchmark/plot_results.py || echo "[warn] 画图失败（可能是某个 CSV 为空）"

# ---------------------------------------------------------------------------
# 4. Task 2：Triton
# ---------------------------------------------------------------------------
hr; echo "[4/6] Task 2：Triton（Vector Add / GEMM / RMSNorm / Autotuning）"
if (cd task2 && "${PY}" triton/bench_all.py); then
  echo "[ok] Task 2 完成"
else
  echo "[warn] Task 2 有失败项，上方日志为准"
fi

# ---------------------------------------------------------------------------
# 5. 打包
# ---------------------------------------------------------------------------
hr; echo "[5/6] 打包结果"
"${PY}" - <<'PYEOF'
import os, zipfile, glob
out = "mlsys_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for base in ("task1/results", "task2/results"):
        for p in glob.glob(base + "/**", recursive=True):
            if os.path.isfile(p):
                zf.write(p, p)
    for extra in glob.glob("task0/env/**", recursive=True):
        if os.path.isfile(extra):
            zf.write(extra, extra)
    for extra in glob.glob("task1/build_and_run.log"):
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
echo "[6/6] 全部结束。下一步："
echo "  1. 把 mlsys_results.zip 从 WSL 拷回 Windows（在 /mnt/d/... 下的话直接资源管理器就能看到）"
echo "  2. 把 zip 或 CSV 发回来，我帮你补齐两份报告里的 [待填]"
echo "  3. 顺手提交回仓库：git add -A && git commit -m '补齐实测数据' && git push"
