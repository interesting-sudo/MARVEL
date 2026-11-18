"""按覆盖阶段分解 Overlap Ratio，定位残余重叠的来源。

用法: python diagnose_phase_overlap.py <off.jsonl> <coord.jsonl> [more.jsonl ...]

输出:
  1) 按"已探索率区间"分箱的 overlap（跨集平均）
  2) 各阶段对总重叠的贡献占比
  3) 按决策步序号分箱（前几步 vs 中段 vs 尾段）
"""
import json
import sys
from collections import defaultdict


def load(path):
    eps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                eps.append(json.loads(line))
    return eps


COV_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]
STEP_BINS = [(0, 3), (3, 8), (8, 20), (20, 10 ** 6)]


def bin_stats(eps, key_hist, bins, bin_name):
    # 每个 episode 内先算各箱均值, 再跨集平均 (等权 episode)
    per_bin = defaultdict(list)
    per_bin_sum = defaultdict(float)
    total_sum = 0.0
    total_n = 0
    for e in eps:
        ov = e["overlap_ratio_history"]
        cov = e["explored_rate_history"]
        acc = defaultdict(list)
        for i, o in enumerate(ov):
            if key_hist == "cov":
                c = cov[i] if i < len(cov) else cov[-1]
                for lo, hi in bins:
                    if lo <= c < hi:
                        acc[(lo, hi)].append(o)
                        break
            else:
                for lo, hi in bins:
                    if lo <= i < hi:
                        acc[(lo, hi)].append(o)
                        break
        for b, vals in acc.items():
            per_bin[b].append(sum(vals) / len(vals))
            per_bin_sum[b] += sum(vals)
        total_sum += sum(ov)
        total_n += len(ov)
    print(f"\n  [{bin_name}] 跨集平均 overlap（括号内=该阶段占总重叠的份额）")
    for b in bins:
        vals = per_bin.get(b, [])
        if not vals:
            print(f"    {b}: 无样本")
            continue
        share = per_bin_sum[b] / total_sum if total_sum else 0.0
        print(f"    {str(b):>16}: overlap={sum(vals)/len(vals):.4f}  share={share*100:5.1f}%  n_ep={len(vals)}")
    return per_bin, total_sum, total_n


def summarize(name, eps):
    ov = [e["overlap_ratio_history"] for e in eps]
    step_mean = sum(sum(o) / len(o) for o in ov) / len(ov)
    print(f"\n=== {name} | episodes={len(eps)} | 步均 overlap={step_mean:.4f} ===")
    cov_freq = defaultdict(int)
    for e in eps:
        for c in e["explored_rate_history"]:
            for lo, hi in COV_BINS:
                if lo <= c < hi:
                    cov_freq[(lo, hi)] += 1
                    break
    tot = sum(cov_freq.values())
    print("  步数分布（按覆盖区间）:", {f"{lo}-{hi}": f"{cov_freq[(lo,hi)]/tot*100:.0f}%" for lo, hi in COV_BINS})
    bin_stats(eps, "cov", COV_BINS, "按覆盖阶段")
    bin_stats(eps, "step", STEP_BINS, "按决策步序号")


def main():
    paths = sys.argv[1:]
    data = [(p, load(p)) for p in paths]
    for p, eps in data:
        summarize(p.split("/")[-1], eps)


if __name__ == "__main__":
    main()
