#!/usr/bin/env python3
"""task4/scripts/quant_check.py

复算 Task 4 手算一：把 x = [-1.0, -0.4, 0.2, 0.9] 分别做
对称 Per-tensor INT8 / INT4 量化，打印 Scale、q、反量化值、绝对/相对误差。

约定（与题面一致）：
    INT8: q ∈ [-127, 127]   INT4: q ∈ [-7, 7]
    s = max(|x|) / q_max,  z = 0
    q = round(x / s) 后 clip 到表示范围； x_hat = s * q
    Round-to-nearest（.5 远离 0）

用法：python3 task4/scripts/quant_check.py
"""
from __future__ import annotations

import math

X = [-1.0, -0.4, 0.2, 0.9]


def round_half_away_from_zero(v: float) -> int:
    """Round-to-nearest，.5 向远离 0 的方向取整。"""
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def quantize(x: list[float], qmax: int) -> dict:
    s = max(abs(v) for v in x) / qmax
    rows = []
    for v in x:
        raw = v / s
        q = max(-qmax, min(qmax, round_half_away_from_zero(raw)))
        x_hat = s * q
        abs_err = abs(x_hat - v)
        rel_err = abs_err / abs(v) if v != 0 else 0.0
        rows.append((v, raw, q, x_hat, abs_err, rel_err))
    return {"scale": s, "qmax": qmax, "rows": rows}


def show(name: str, qmax: int) -> None:
    res = quantize(X, qmax)
    s = res["scale"]
    print(f"=== {name} (q ∈ [-{qmax}, {qmax}]) ===")
    print(f"s = max|x| / q_max = 1.0 / {qmax} = {s:.10f}")
    print(f"{'x':>7} {'x/s':>10} {'q':>6} {'x_hat':>14} {'abs_err':>12} {'rel_err':>9}")
    for v, raw, q, x_hat, abs_err, rel_err in res["rows"]:
        print(f"{v:>7.2f} {raw:>10.4f} {q:>6d} {x_hat:>14.8f} {abs_err:>12.8f} {rel_err*100:>8.2f}%")
    worst_abs = max(r[4] for r in res["rows"])
    worst_rel = max(res["rows"], key=lambda r: r[5])
    print(f"max abs err = {worst_abs:.8f}")
    print(f"max rel err = {worst_rel[5]*100:.2f}%  (发生在 x = {worst_rel[0]})")
    print()


def main() -> None:
    show("INT8 对称 Per-tensor", 127)
    show("INT4 对称 Per-tensor", 7)

    print("结论：")
    print("1) |x| 最大的元素 -1.0 恰好映射到 q_max，可被精确表示；")
    print("2) 其余元素按步长 s 量化，绝对误差上界为 s/2；")
    print("3) Bit 数减少后，最先失去精度的是**绝对值最小的非零元素**：")
    print("   INT8 里 0.2 的相对误差约 1.57%，INT4 里升到约 28.57%；")
    print("   因为相对误差 ~ (s/2)/|x|，s 变大约 18 倍（127/7），小值最先崩。")


if __name__ == "__main__":
    main()
