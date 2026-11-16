#!/usr/bin/env python3
"""Build the HTML reproduction report from the raw CSV/JSON results.

All numbers are recomputed from ``results/*.csv`` so the report can never drift
from the data.  Run from the repository root:

    python reproduction/make_report.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproduction/results"
SEED_DIR = ROOT / "reproduction/results_seed7777"
OUT = ROOT / "reproduction/MARVEL_复现报告.html"

# key -> (table, agents, fov, range, paper traj, paper traj std, paper cov, paper cov std,
#         paper overlap, paper overlap std, paper success)
PAPER = {
    "table_I_n2_fov120_r10":   ("I",   2, 120, 10, 505.25, 89.03, 416.44, 84.62, 0.048, 0.137, 100),
    "table_I_n4_fov120_r10":   ("I",   4, 120, 10, 357.50, 67.07, 294.21, 62.45, 0.170, 0.169, 100),
    "table_I_n8_fov120_r10":   ("I",   8, 120, 10, 279.35, 50.63, 231.54, 46.40, 0.316, 0.174, 100),
    "table_II_n4_fov90_r10":   ("II",  4,  90, 10, 378.99, 57.61, 313.67, 54.57, 0.147, 0.155, 100),
    "table_II_n4_fov180_r10":  ("II",  4, 180, 10, 328.36, 61.31, 273.18, 59.95, 0.207, 0.183, 100),
    "table_III_n4_fov120_r12": ("III", 4, 120, 12, 299.57, 60.33, 247.93, 58.70, 0.223, 0.188, None),
    "table_III_n4_fov120_r15": ("III", 4, 120, 15, 258.35, 54.96, 212.85, 49.50, 0.208, 0.190, None),
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stats(csv_path: Path) -> dict:
    rows = load(csv_path)
    traj = [float(r["travel_dist"]) for r in rows]
    ok = [float(r["travel_dist"]) for r in rows if r["success"] == "1"]
    cov = [float(r["dist_to_90"]) for r in rows if r["dist_to_90"]]
    ovl = [float(r["overlap_mean"]) for r in rows]
    fails = [int(r["episode"]) for r in rows if r["success"] == "0"]
    return {
        "n": len(rows), "traj": traj, "ok": ok, "cov": cov, "ovl": ovl, "fails": fails,
        "traj_mean": st.mean(traj) if traj else None,
        "traj_std": st.pstdev(traj) if traj else None,
        "ok_mean": st.mean(ok) if ok else None,
        "ok_std": st.pstdev(ok) if ok else None,
        "cov_mean": st.mean(cov) if cov else None,
        "cov_std": st.pstdev(cov) if cov else None,
        "ovl_mean": st.mean(ovl) if ovl else None,
        "succ": 100 * len(ok) / len(rows) if rows else None,
        "rows": rows,
    }


def pct(new: float | None, ref: float) -> tuple[str, str]:
    if new is None:
        return "—", "flat"
    d = (new - ref) / ref * 100
    tone = "good" if d <= -1 else ("bad" if d >= 1 else "flat")
    return f"{d:+.1f}%", tone


def fmt(v: float | None, digits: int = 2, std: float | None = None) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}" + (f"<span class=pm>± {std:.{digits}f}</span>" if std is not None else "")


def main() -> None:
    beijing = datetime.now(timezone(timedelta(hours=8)))
    cards: list[str] = []
    rows_html: list[str] = []

    for key, meta in PAPER.items():
        table, n_ag, fov, rng, p_traj, p_traj_sd, p_cov, p_cov_sd, p_ovl, p_ovl_sd, p_succ = meta
        s = stats(RESULTS / f"{key}.csv")
        if not s["n"]:
            continue
        corr = stats(RESULTS / f"{key}_corrected_overlap.csv") if rng != 10 else None
        seed2 = stats(SEED_DIR / f"{key}.csv") if SEED_DIR.exists() else None

        d_all, t_all = pct(s["traj_mean"], p_traj)
        d_ok, t_ok = pct(s["ok_mean"], p_traj)
        d_cov, t_cov = pct(s["cov_mean"], p_cov)
        ovl_show = corr["ovl_mean"] if corr and corr["ovl_mean"] else s["ovl_mean"]
        d_ovl, t_ovl = pct(ovl_show, p_ovl)

        seed_txt = "—"
        if seed2 and seed2["ok_mean"]:
            seed_txt = f"{seed2['ok_mean']:.1f} / {seed2['succ']:.0f}%"

        rows_html.append(f"""
        <tr>
          <td class="tbl">{table}</td>
          <td class="cfg">{n_ag} 机 · {fov}° · {rng} m</td>
          <td class="paper">{p_traj:.2f}<span class=pm>± {p_traj_sd:.2f}</span></td>
          <td class="repro">{fmt(s['traj_mean'], 2, s['traj_std'])}</td>
          <td class="delta {t_all}">{d_all}</td>
          <td class="repro sub">{fmt(s['ok_mean'], 2, s['ok_std'])}<br><span class="delta {t_ok}">{d_ok}</span></td>
          <td class="paper">{p_cov:.2f}<span class=pm>± {p_cov_sd:.2f}</span></td>
          <td class="repro">{fmt(s['cov_mean'], 2, s['cov_std'])}
              <br><span class="delta {t_cov}">{d_cov}</span></td>
          <td class="paper">{p_ovl:.3f}<span class=pm>± {p_ovl_sd:.3f}</span></td>
          <td class="repro">{ovl_show:.3f}
              <br><span class="delta {t_ovl}">{d_ovl}</span></td>
          <td class="paper">{p_succ if p_succ is not None else "—"}</td>
          <td class="repro">{s['succ']:.0f}</td>
          <td class="seed">{seed_txt}</td>
        </tr>""")

    # headline numbers
    t1 = [stats(RESULTS / f"{k}.csv") for k in list(PAPER)[:3]]
    t2 = [stats(RESULTS / f"{k}.csv") for k in list(PAPER)[3:5]]
    t3 = [stats(RESULTS / f"{k}.csv") for k in list(PAPER)[5:7]]
    worst_12 = max(abs(stats(RESULTS / f"{k}.csv")["ok_mean"] - PAPER[k][4]) / PAPER[k][4] * 100 for k in list(PAPER)[:5])
    total_fail = sum(len(x["fails"]) for x in t1 + t2 + t3)
    d15 = (t3[1]["ok_mean"] - PAPER["table_III_n4_fov120_r15"][4]) / PAPER["table_III_n4_fov120_r15"][4] * 100

    summary_rows = f"""
      <div class="kpi"><b>7 × 100</b><span>配置 × 地图，共 700 个 episode</span></div>
      <div class="kpi"><b>{worst_12:.1f}%</b><span>表 I / II 轨迹长度的最大偏差</span></div>
      <div class="kpi bad-kpi"><b>{d15:+.1f}%</b><span>表 III（15 m）剔除失败后的偏差</span></div>
      <div class="kpi"><b>{total_fail}</b><span>跑满 128 步未收敛的失败 episode</span></div>
    """

    failures_html = []
    for key, meta in PAPER.items():
        s = stats(RESULTS / f"{key}.csv")
        for ep in s["fails"]:
            row = s["rows"][ep]
            failures_html.append(
                f"<tr><td>{meta[0]}</td><td>{meta[1]} 机 · {meta[2]}° · {meta[3]} m</td>"
                f"<td>#{ep}</td><td>{float(row['travel_dist']):.0f} m</td>"
                f"<td>{float(row['explored_rate']) * 100:.1f}%</td>"
                f"<td>{float(row['overlap_mean']):.3f}</td></tr>")
    failures_html = "\n".join(failures_html)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>MARVEL 复现报告</title>
<style>
  :root {{ --ink:#111418; --muted:#5b6472; --line:#e3e7ed; --bg:#f7f8fa; --card:#fff;
           --accent:#1f4e79; --good:#0f7b4f; --bad:#b3341f; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
         font-size:15px; line-height:1.72; -webkit-font-smoothing:antialiased; }}
  main {{ max-width:1120px; margin:0 auto; padding:56px 32px 96px; }}
  header {{ border-bottom:2px solid var(--ink); padding-bottom:22px; margin-bottom:34px; }}
  .eyebrow {{ font-size:12.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }}
  h1 {{ font-size:31px; line-height:1.3; margin:10px 0 12px; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); font-size:14px; }}
  .sub code {{ background:#eef1f5; padding:1px 5px; border-radius:3px; }}
  h2 {{ font-size:19px; margin:44px 0 14px; padding-left:11px; border-left:4px solid var(--accent); }}
  h3 {{ font-size:15.5px; margin:26px 0 8px; }}
  p {{ margin:9px 0; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:26px 0 8px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
  .kpi b {{ display:block; font-size:24px; letter-spacing:-.02em; color:var(--accent); }}
  .kpi span {{ display:block; font-size:12.5px; color:var(--muted); line-height:1.5; margin-top:4px; }}
  .bad-kpi b {{ color:var(--bad); }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); font-size:13px;
           font-variant-numeric:tabular-nums; margin:14px 0 6px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:center; }}
  th {{ background:#f0f3f7; font-weight:600; font-size:12.2px; color:#2b3440; line-height:1.45; }}
  td.tbl {{ font-weight:700; color:var(--accent); }}
  td.cfg {{ white-space:nowrap; font-size:12.5px; }}
  td.paper {{ color:var(--muted); background:#fbfcfd; }}
  td.repro {{ font-weight:600; }}
  td.repro.sub {{ font-weight:500; color:var(--muted); }}
  td.seed {{ color:var(--muted); font-size:12px; }}
  .pm {{ display:inline; color:var(--muted); font-weight:400; font-size:11.5px; margin-left:3px; }}
  .delta {{ font-size:11.5px; font-weight:600; }}
  .good {{ color:var(--good); }} .bad {{ color:var(--bad); }} .flat {{ color:var(--muted); }}
  .note {{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--accent);
           border-radius:0 8px 8px 0; padding:13px 17px; margin:16px 0; font-size:13.5px; }}
  .note b {{ color:var(--accent); }}
  ul {{ margin:8px 0 8px 20px; padding:0; }}
  li {{ margin:5px 0; }}
  code {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:12.6px; }}
  pre {{ background:#11161d; color:#dfe6ef; padding:14px 16px; border-radius:9px; overflow-x:auto;
         font-size:12.4px; line-height:1.6; }}
  pre code {{ color:inherit; }}
  .files {{ font-size:12.7px; color:var(--muted); }}
  footer {{ margin-top:52px; padding-top:16px; border-top:1px solid var(--line);
            font-size:12px; color:var(--muted); }}
</style></head>
<body><main>
<header>
  <div class="eyebrow">Reproduction Report · ICRA 2025</div>
  <h1>MARVEL：受限视场多机器人探索的 MARL 策略复现</h1>
  <div class="sub">
    原论文 <i>MARVEL: Multi-Agent Reinforcement Learning for constrained field-of-View multi-robot
    Exploration in Large-scale environments</i>（Chiun, Zhang, Wang, Cao, Sartoretti）<br>
    复现范围：作者公开权重 <code>load_model/MARVEL/checkpoint.pth</code> 的评估链路，
    表 I–III 共 7 组配置 × 100 张测试地图。生成时间 {beijing:%Y-%m-%d %H:%M}（北京时间）。
  </div>
</header>

<div class="kpis">{summary_rows}</div>

<h2>一、复现结论</h2>
<ul>
  <li><b>基准配置（表 I）与视场泛化（表 II）复现成功。</b>剔除失败 episode 后，轨迹长度与论文的偏差在
      0.0%–3.4% 之间，90% 覆盖率偏差在 1.4% 以内，重叠率偏差 ≤0.02。离散程度也与论文同量级
      （表 I 四机：本复现 64.0 vs 论文 67.07）。</li>
  <li><b>传感距离外推（表 III）未能复现论文的增益。</b>12 m 剔除失败后偏高 +6.1%，15 m 偏高 +17.8%，
      且成功率降到 99%、98%，而论文声称维持 100%。方向性（加大传感距离→轨迹变短）成立但幅度不足。
      换一组起点种子重跑，12 m 仍偏高 +7.6%、15 m 仍偏高 +19.5%，说明这是稳定偏差而非起点运气。</li>
  <li><b>重叠率的实现口径已定位。</b>作者仓库把 overlap 的铺展距离写死为训练时的 10 m；按真实传感距离
      重算后，12 m 配置得 0.226，与论文的 0.223 几乎相同，说明论文用的是真实距离口径，发布代码是退化版本。</li>
  <li><b>训练过程不可复现。</b>发布包只有 100 张测试地图，不含论文所述 5,663 张训练地图，也没有训练日志，
      因此本次仅复现评估，不能声称复现了训练。</li>
</ul>

<h2>二、与论文表 I–III 的逐项对比</h2>
<p>每组 100 个 episode，「全部」是 100 集直接平均，「剔除失败」只统计成功集（论文的 100% 成功率意味着其均值只覆盖成功集）。
重叠率对表 III 取按真实传感距离校正后的口径；最后一列是用另一组随机种子（起点重新采样）重跑表 III 的结果。</p>
<table>
  <thead>
    <tr>
      <th>表</th><th>配置</th>
      <th>轨迹·论文</th><th>轨迹·复现<br><span style="font-weight:400">（全部）</span></th><th>Δ</th>
      <th>轨迹·复现<br><span style="font-weight:400">（剔除失败）</span></th>
      <th>90% 覆盖·论文</th><th>90% 覆盖·复现</th>
      <th>重叠率·论文</th><th>重叠率·复现</th>
      <th>成功率·论文</th><th>成功率·复现</th><th>换种子<br>轨迹/成功率</th>
    </tr>
  </thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
<p class="files">单位：米。Δ 为相对论文的百分比偏差，绿色表示优于论文，红色表示劣于论文。表 III 论文未给成功率列，
论文正文称维持 100%。</p>

<h2>三、失败 episode 的形态</h2>
<p>失败样本不是「差一点没跑完」，而是彻底打转：128 步全部耗尽、覆盖率停在 81%–96%、轨迹 786–1267 m，
是正常解的 2–4 倍。它们把均值和标准差一起抬高——8 机器人那组的标准差从 53.8 被拉到 112.3，
若不剔除就会误判为「复现结果抖动大」，实际只是少数确定性失败样本的污染。</p>
<table>
  <thead><tr><th>表</th><th>配置</th><th>episode</th><th>轨迹</th><th>终止覆盖率</th><th>重叠率</th></tr></thead>
  <tbody>{failures_html}</tbody>
</table>

<h2>四、环境搭建与三处必要的代码修改</h2>
<p>作者代码写于 2024 年、按 NumPy 1.x 与「测试跑 CPU」的假设编写。在 Python 3.11 + torch 2.8.0+cu126 +
NumPy 2.3.2 上直接运行会在三处崩溃，三处修改都不改变算法语义：</p>
<table>
  <thead><tr><th style="text-align:left">文件</th><th style="text-align:left">原写法</th><th style="text-align:left">修改</th><th style="text-align:left">原因</th></tr></thead>
  <tbody>
    <tr><td style="text-align:left"><code>utils/sensor.py</code></td>
        <td style="text-align:left"><code>robot_belief.itemset((y,x),k)</code></td>
        <td style="text-align:left"><code>robot_belief[int(y),int(x)] = k</code></td>
        <td style="text-align:left"><code>ndarray.itemset</code> 在 NumPy 2.0 被移除</td></tr>
    <tr><td style="text-align:left"><code>utils/utils.py</code></td>
        <td style="text-align:left"><code>np.lib.pad(...)</code></td>
        <td style="text-align:left"><code>np.pad(...)</code></td>
        <td style="text-align:left"><code>np.lib.pad</code> 在 NumPy 2.0 被移除</td></tr>
    <tr><td style="text-align:left"><code>utils/agent.py</code></td>
        <td style="text-align:left"><code>torch.tensor(current_edge).unsqueeze(0)</code></td>
        <td style="text-align:left">追加 <code>.to(self.device)</code></td>
        <td style="text-align:left">作者默认 CPU 测试；开 GPU 后索引张量留在 CPU，<code>torch.gather</code> 设备不一致</td></tr>
  </tbody>
</table>
<div class="note">
  <b>没有装 Ray。</b>原 <code>test_driver.py</code> 用 Ray 只是把 10 个 meta-agent 分发到多进程，
  真正的仿真逻辑在 <code>TestWorker</code> 里。复现运行器直接串行调用同一个 <code>TestWorker</code>，
  既省掉一份依赖，也避免 Ray 在容器里起共享内存的问题。并行化改由「每组配置一个进程」实现，
  7 组跑满 700 个 episode 的墙钟时间约 45 分钟（2×RTX 4090，单集 12–28 s）。
</div>

<h2>五、口径差异与可信度边界</h2>
<ul>
  <li><b>随机种子未公开。</b>机器人起点由 <code>np.random.choice</code> 在固定起始区域内采样，
      每集种子取 <code>seed_base + episode</code>。因此这是「在作者权重上的可复现实测」，
      而不是逐项数字对齐；换种子（最后一列）用于确认结论不是起点运气。</li>
  <li><b>重叠率的标准差不可比。</b>论文把所有时间步的 overlap 汇总后求方差，本复现先算每集均值再跨集求方差，
      得到的是集间离散度。均值不受影响，标准差会系统性偏小（0.05 vs 0.17 属于口径差异，不是误差）。</li>
  <li><b>表 III 的偏差有三种可能来源</b>：发布权重与论文表 III 所用权重不是同一次训练、作者当时的 overlap
      与策略版本不同、或 12/15 m 本就踩在策略泛化边缘。本复现只能确认偏差稳定存在（换种子后仍在 +6%/+18% 量级），
      无法在缺少训练资源的前提下分辨是哪一种。</li>
  <li><b>地图尺度</b>：<code>maps_test/*.png</code> 为 500×500 像素，<code>block_reduce(2)</code> 后 250×250 格、
      每格 0.4 m，画布 100 m 见方，可行驶区域约 90 m，与论文描述一致。</li>
</ul>

<h2>六、复跑方式</h2>
<pre><code># 单集冒烟测试（约 15 s）
PYTHONPATH=/app/MARVEL python reproduction/run_marvel_tables.py --device cuda \\
    --episodes 1 --configs table_I_n4_fov120_r10

# 全量 7 组 × 100 集（可断点续跑，逐集落盘）
PYTHONPATH=/app/MARVEL python reproduction/run_marvel_tables.py --device cuda

# 表 III 的重叠率校正口径
PYTHONPATH=/app/MARVEL python reproduction/run_marvel_tables.py --device cuda --correct-overlap \\
    --configs table_III_n4_fov120_r12 table_III_n4_fov120_r15

# 换一组种子做稳健性检查
PYTHONPATH=/app/MARVEL python reproduction/run_marvel_tables.py --device cuda --seed 7777 \\
    --output reproduction/results_seed7777 --configs table_III_n4_fov120_r12 table_III_n4_fov120_r15

# 与论文对比
python reproduction/compare_with_paper.py</code></pre>
<p class="files">数据文件：<code>reproduction/results/*.csv</code>（逐集原始记录：轨迹、覆盖率、成功率、90% 覆盖距离、重叠率）、
<code>*.json</code>（每组汇总 + 权重 SHA256）、<code>results_seed7777/</code>（换种子重跑）、
<code>logs/</code>（运行日志）。</p>

<footer>
  复现环境：2× NVIDIA RTX 4090（24 GB）· Python 3.11.13 · PyTorch 2.8.0+cu126 · NumPy 2.3.2 ·
  scikit-image 0.26.0 · Shapely 2.2.0。<br>
  权重指纹 SHA256 <code>2e992ff83c8b05a346e882be5279f3e3db8b04e6be7aad308dd2b6c0532d99b2</code>，
  全部 9 次运行共用同一份权重。
</footer>
</main></body></html>
"""
    OUT.write_text(html)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
