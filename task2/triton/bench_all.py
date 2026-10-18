"""task2/triton/bench_all.py

一键跑完 Task 2 的全部实验（Vector Add / GEMM / Autotuning），
结果写入 ../results/*.json。

用法：
    cd task2 && python3 triton/bench_all.py
"""
from __future__ import annotations

import importlib


def main() -> None:
    for module_name in ("vector_add", "gemm", "autotune_search"):
        print("=" * 78)
        print(f"# {module_name}")
        print("=" * 78)
        module = importlib.import_module(module_name)
        module.main()


if __name__ == "__main__":
    main()
