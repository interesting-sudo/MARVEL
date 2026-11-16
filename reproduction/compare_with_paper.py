#!/usr/bin/env python3
"""Compare reproduced MARVEL metrics against the paper's Tables I--III.

Reads the *.json summaries written by run_marvel_tables.py and prints a
markdown table with the paper value, the reproduced value and the relative
deviation.  Paper numbers are transcribed from the ICRA 2025 camera-ready PDF
(pages 5-6).
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# (name, table, paper trajectory, paper 90% coverage, paper overlap, paper success %)
PAPER = {
    "table_I_n2_fov120_r10":   ("I",   (505.25, 89.03), (416.44, 84.62), (0.048, 0.137), 100),
    "table_I_n4_fov120_r10":   ("I",   (357.50, 67.07), (294.21, 62.45), (0.170, 0.169), 100),
    "table_I_n8_fov120_r10":   ("I",   (279.35, 50.63), (231.54, 46.40), (0.316, 0.174), 100),
    "table_II_n4_fov90_r10":   ("II",  (378.99, 57.61), (313.67, 54.57), (0.147, 0.155), 100),
    "table_II_n4_fov180_r10":  ("II",  (328.36, 61.31), (273.18, 59.95), (0.207, 0.183), 100),
    "table_III_n4_fov120_r12": ("III", (299.57, 60.33), (247.93, 58.70), (0.223, 0.188), None),
    "table_III_n4_fov120_r15": ("III", (258.35, 54.96), (212.85, 49.50), (0.208, 0.190), None),
}

HEADER = (
    "| Table | Config | 轨迹长度 (m) 论文 | 复现 | Δ | 90% 覆盖 (m) 论文 | 复现 | Δ | "
    "重叠率 论文 | 复现 | 成功率 论文 | 复现 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def fmt(value: float, std: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")


def delta(reproduced: float, paper: float) -> str:
    if reproduced is None:
        return "—"
    return f"{(reproduced - paper) / paper * 100:+.1f}%"


def main() -> None:
    rows = []
    for name, (table, traj, cov, ovl, succ) in PAPER.items():
        path = RESULTS / f"{name}.json"
        if not path.exists():
            rows.append(f"| {table} | {name} | 未运行 | | | | | | | | | |")
            continue
        data = json.loads(path.read_text())
        t, t_std = data["trajectory_mean_std"]
        c, c_std = data["coverage90_mean_std"]
        o, o_std = data["overlap_mean_std"]
        s = data["success_percent"]
        rows.append(
            f"| {table} | {name.replace('table_', '')} "
            f"| {fmt(*traj)} | {fmt(t, t_std)} | {delta(t, traj[0])} "
            f"| {fmt(*cov)} | {fmt(c, c_std)} | {delta(c, cov[0])} "
            f"| {fmt(ovl[0], ovl[1], 3)} | {fmt(o, o_std, 3)} "
            f"| {succ if succ is not None else '—'} | {s:.0f} |"
        )
    print(HEADER + "\n".join(rows))


if __name__ == "__main__":
    main()
