"""探测起始位置分布：8 机是否起点聚在一起，以及 t=0 重叠的几何下限。"""
import numpy as np
from utils.env_test import Env
from itertools import combinations

SEED = 20261118
N = 8
rows = []
for ep in range(20):
    np.random.seed(SEED + ep)
    env = Env(ep, 120, N, 10)
    P = np.array(env.robot_locations, dtype=float)
    d = [np.linalg.norm(P[i] - P[j]) for i, j in combinations(range(N), 2)]
    rows.append((ep, np.min(d), np.mean(d), np.max(d), np.std(P[:, 0]) + np.std(P[:, 1])))

print(f"{'ep':>3} {'min_pair':>9} {'mean_pair':>9} {'max_pair':>9} {'spread':>7}")
for r in rows:
    print(f"{r[0]:>3} {r[1]:9.2f} {r[2]:9.2f} {r[3]:9.2f} {r[4]:7.2f}")
arr = np.array([r[1:] for r in rows])
print("\n平均: min_pair=%.2f mean_pair=%.2f max_pair=%.2f spread=%.2f  (传感 10 m)" % tuple(arr.mean(0)))
print("min_pair < 10 m 的集数: %d/20;  min_pair < 5 m: %d/20" % (np.sum(arr[:, 0] < 10), np.sum(arr[:, 0] < 5)))
