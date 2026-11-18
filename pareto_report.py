"""把多个配置放在同一张 Pareto 表里，同时给出相对 MARVEL 基线和 MMPF 的差距。

用法:
    python pareto_report.py results/overlap_coord
"""
import json
import os
import sys
import glob

# MMPF 论文 Table I 的参考值（8 机 / 4 机），来源: 重叠协调改进报告.md 表 8-1 与第九节
MMPF = {8: dict(overlap=0.172, travel=311.21, d90=279.76),
        4: dict(overlap=0.075, travel=427.30, d90=358.56)}


def load(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stats(rows, path=None):
    n = len(rows)
    overlap = sum(sum(r['overlap_ratio_history']) / max(len(r['overlap_ratio_history']), 1)
                  for r in rows) / n
    steady = sum(sum(r['overlap_ratio_history'][20:]) / max(len(r['overlap_ratio_history'][20:]), 1)
                 for r in rows) / n
    travel = sum(r['travel_dist'] for r in rows) / n
    # 未达到 90% 覆盖的集会把这些字段写成逐步列表，统计时排除
    valid_d90 = [r['dist_to_0_90'] for r in rows if isinstance(r['dist_to_0_90'], (int, float))]
    d90 = sum(valid_d90) / max(len(valid_d90), 1)
    succ = sum(1 for r in rows if r['success_rate']) / n
    steps = sum(len(r['overlap_ratio_history']) for r in rows) / n
    explored = sum(r.get('explored_rate', 0.0) for r in rows) / n
    result = dict(n=n, overlap=overlap, steady=steady, travel=travel, d90=d90,
                  succ=succ, steps=steps, explored=explored, n_d90=len(valid_d90))
    # summary.json 的均值与论文口径一致，优先采用
    if path:
        summary = os.path.splitext(path)[0] + '.summary.json'
        if os.path.exists(summary):
            with open(summary) as handle:
                s = json.load(handle)
            for key, name in (('overlap', 'overlap_ratio_mean'), ('travel', 'travel_dist_mean'),
                              ('d90', 'dist_to_0_90_mean'), ('succ', 'success_rate'),
                              ('explored', 'explored_rate_mean')):
                if name in s:
                    result[key] = s[name]
    return result


def n_agents_of(path):
    summary = os.path.splitext(path)[0] + '.summary.json'
    if os.path.exists(summary):
        with open(summary) as handle:
            return json.load(handle).get('config', {}).get('n_agents', 0)
    return 0


def main(root):
    files = sorted(glob.glob(os.path.join(root, '**', '*.jsonl'), recursive=True))
    entries = []
    for path in files:
        rows = load(path)
        if not rows:
            continue
        entries.append((os.path.relpath(path, root), stats(rows, path), n_agents_of(path)))

    print('| 配置 | 机数 | 集 | Overlap | 稳态(≥20步) | 轨迹 m | Dist@90 m | 成功率 | 步数 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name, s, n in sorted(entries, key=lambda e: (e[2], e[1]['overlap'])):
        print('| {} | {} | {} | {:.4f} | {:.4f} | {:.1f} | {:.1f} | {:.0f}% | {:.1f} |'.format(
            name, n, s['n'], s['overlap'], s['steady'], s['travel'], s['d90'],
            100 * s['succ'], s['steps']))

    print()
    print('相对同机数 off 基线的变化（%%）：')
    baseline = {}
    for name, s, n in entries:
        if 'off' in os.path.basename(name):
            baseline.setdefault(n, s)
    print('| 配置 | ΔOverlap | Δ轨迹 | ΔDist@90 | 对 MMPF 重叠 | 对 MMPF 轨迹 |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, s, n in sorted(entries, key=lambda e: (e[2], e[1]['overlap'])):
        base = baseline.get(n)
        if base is None:
            continue
        mmpf = MMPF.get(n)
        dovl = 100 * (s['overlap'] - base['overlap']) / base['overlap']
        dtr = 100 * (s['travel'] - base['travel']) / base['travel']
        dd90 = 100 * (s['d90'] - base['d90']) / base['d90']
        if mmpf:
            m1 = '{:+.1f}%'.format(100 * (s['overlap'] - mmpf['overlap']) / mmpf['overlap'])
            m2 = '{:+.1f}%'.format(100 * (s['travel'] - mmpf['travel']) / mmpf['travel'])
        else:
            m1 = m2 = '—'
        print('| {} | {:+.1f}% | {:+.1f}% | {:+.1f}% | {} | {} |'.format(
            name, dovl, dtr, dd90, m1, m2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/overlap_coord')
