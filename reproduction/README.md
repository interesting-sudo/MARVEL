# MARVEL（ICRA 2025）表 I–III 权重复现

在作者仓库根目录运行。运行器加载 `load_model/MARVEL/checkpoint.pth`，不训练模型，按 7 组配置各评估 `maps_test` 中的 100 个 episode，并将每个 episode 的结果立即写入 CSV，因此可安全地重启。

```bash
PYTHONPATH=. python reproduction/run_marvel_tables.py --device cuda
```

先做一集 GPU 冒烟测试：

```bash
PYTHONPATH=. python reproduction/run_marvel_tables.py --device cuda --episodes 1 --configs table_I_n4_fov120_r10
```

与论文表 I–III 对比：

```bash
python reproduction/compare_with_paper.py
```

论文原代码的 overlap 计算固定使用训练时的 10 m 传感范围（`utils/test_worker.py` 里用的是 `parameter.SENSOR_RANGE`，不是 `self.sensor_range`）。表 III 的轨迹、90% 覆盖率、成功率不受影响；为测量这个实现差异，可在原始口径跑完后再运行：

```bash
PYTHONPATH=. python reproduction/run_marvel_tables.py --device cuda --configs table_III_n4_fov120_r12 table_III_n4_fov120_r15 --correct-overlap
```

## 环境与必要的兼容性修改

实测环境：Python 3.11 + torch 2.8.0+cu126 + numpy 2.3.2 + scikit-image 0.26 + shapely 2.2（作者代码写于 2024 年，按 numpy 1.x 编写，直接跑会在三处崩溃）。三处修改均不改变算法语义：

| 文件 | 原代码 | 修改 | 原因 |
|---|---|---|---|
| `utils/sensor.py` | `robot_belief.itemset((y, x), k)` | `robot_belief[int(y), int(x)] = k` | `ndarray.itemset` 在 NumPy 2.0 被移除 |
| `utils/utils.py` | `np.lib.pad(...)` | `np.pad(...)` | `np.lib.pad` 在 NumPy 2.0 被移除 |
| `utils/agent.py` | `current_edge = torch.tensor(current_edge).unsqueeze(0)` | 末尾加 `.to(self.device)` | 作者默认 `USE_GPU=False` 用 CPU 测试；开 GPU 后索引张量留在 CPU，`torch.gather` 报设备不一致 |

原文件备份为 `*.py.orig`。Ray 不必安装：`test_driver.py` 只是用 Ray 做多进程外壳，运行器直接串行调用同一个 `TestWorker`。

## 复现结果的读法

运行器固定 `maps_test` 文件名排序以及每个 episode 的随机种子（`seed_base + episode`）。论文没有公开种子，机器人起点由 `np.random.choice` 在起始区域随机采样，因此结果应被报告为「在作者权重上的可复现实测」，不能声称逐项数值必然与论文完全相同。

两处口径差异需要说明：

1. **失败集**：论文把成功率写到 100%，其均值只覆盖成功集。本次有若干 episode 跑满 128 步仍未达 99% 覆盖率（轨迹 786–1267 m 的「打转」样本），会把均值与标准差一起抬高。报告里同时给「全部」和「剔除失败」两列。
2. **重叠率**：论文把所有时间步的 overlap 汇总求均值与标准差；运行器先算每集均值再跨集平均，因此标准差是**集间**方差，与论文的**时间步间**方差不可直接比较。均值不受影响。

原始口径与校正口径的结果写在不同文件里，避免混淆。
