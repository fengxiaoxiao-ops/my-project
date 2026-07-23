# -*- coding: utf-8 -*-
"""
武汉工程大学校园交通车问题——完整求解代码
问题一：最少站点覆盖（最小支配集，ILP精确求解，覆盖半径200m）
问题二：站点环线设计（TSP，小规模穷举求最优环线）
问题三：站点选择与环线里程联合优化（枚举多个最小覆盖方案，比较环线里程）
问题四：便利店选址（2-中位设施选址模型，按建筑人流加权）

依赖：numpy, scipy>=1.9 (需要 scipy.optimize.milp)
"""

import numpy as np
from itertools import permutations, combinations
from scipy.optimize import milp, LinearConstraint, Bounds

# ========================================================================
# 0. 数据：建筑物经纬度 -> 平面坐标（米）
# ========================================================================
raw = {
    '育化门(东门)':    (114.432492, 30.46158), '和光门(西一门)':  (114.43055, 30.46092),
    '霞光门(西门)':    (114.42981, 30.46013), '南校门':          (114.43216, 30.45805),
    '一教(中和楼)':    (114.43375, 30.46028), '二教(明理楼)':    (114.43301, 30.46006),
    '三教(致知楼)':    (114.43232, 30.45992), '四教(博闻楼)':    (114.43170, 30.46061),
    '自强楼':          (114.43198, 30.46135), '大化工楼ABC':     (114.43022, 30.46188),
    '1号实验楼(格物楼)': (114.43278, 30.45965), '2号实验楼(安澜楼)': (114.43211, 30.45949),
    '3/4号实验楼(笃行楼)': (114.43136, 30.45933), '创新创业大楼':    (114.43455, 30.46211),
    '主图书馆':        (114.43311, 30.46071), '明志楼':          (114.43105, 30.46166),
    '明善楼':          (114.43068, 30.46139), '求是楼':          (114.43030, 30.46112),
    '教辅1栋':         (114.43092, 30.45901), '教辅2栋':         (114.43133, 30.45915),
    '教辅3栋':         (114.43156, 30.45882), '教辅4/5栋':       (114.43051, 30.45876),
    '教辅6栋':         (114.43078, 30.45838), '教辅7栋':         (114.43025, 30.45822),
    '教辅8栋':         (114.42986, 30.45841), '教辅9栋':         (114.43044, 30.45938),
    '研究生公寓':      (114.42947, 30.45906), '南区18-21栋':     (114.43122, 30.45748),
    '泰塑1-8栋':       (114.43245, 30.45702), '静心湖中心':      (114.43260, 30.45977),
    '大鹏广场':        (114.43215, 30.46102), '北区田径场':      (114.43418, 30.46079),
    '三食堂':          (114.43244, 30.45921), '四食堂':          (114.43062, 30.45929),
    '芳草地快递站':    (114.43433, 30.46127),
}
names = list(raw.keys())
n = len(names)
lonlat = np.array([raw[k] for k in names])

R = 6371000.0
lat0 = np.radians(lonlat[:, 1].mean())
x = R * np.radians(lonlat[:, 0]) * np.cos(lat0)
y = R * np.radians(lonlat[:, 1])
x -= x.min(); y -= y.min()
coords = np.column_stack([x, y])

dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist[i, j] = np.linalg.norm(coords[i] - coords[j])


# ========================================================================
# 1. 问题一：最小覆盖站点选择（最小支配集 ILP）
#    要求：非站点建筑到最近站点距离 <= 200m；目标：站点数最少
# ========================================================================
def solve_min_cover(radius=200, forbidden_sets=None):
    cover = (dist <= radius).astype(float)
    A = cover.T
    c = np.ones(n)
    constraints = [LinearConstraint(A, lb=1, ub=np.inf)]
    if forbidden_sets:
        for fs in forbidden_sets:
            row = np.zeros(n); row[list(fs)] = 1
            constraints.append(LinearConstraint(row, lb=-np.inf, ub=len(fs) - 1))
    res = milp(c, constraints=constraints, bounds=Bounds(0, 1), integrality=np.ones(n))
    if res.x is None:
        return None
    return tuple(np.where(res.x > 0.5)[0])

def q1():
    selected = solve_min_cover()
    print(f"===== 问题一：最少站点覆盖 =====")
    print(f"最少站点数: {len(selected)}")
    print("站点: " + ", ".join(names[i] for i in selected))
    max_d = max(dist[j, selected].min() for j in range(n) if j not in selected)
    print(f"未设站建筑到最近站点的最大距离: {max_d:.1f} m (约束: <=200m)\n")
    return selected


# ========================================================================
# 2. 问题二：站点环线设计（站点数少，直接穷举全排列求最优TSP闭环）
# ========================================================================
def loop_length(idxs, order):
    seq = [idxs[i] for i in order] + [idxs[order[0]]]
    return sum(dist[seq[i], seq[i+1]] for i in range(len(seq) - 1))

def best_loop(idxs):
    k = len(idxs)
    best_len, best_order = np.inf, None
    for perm in permutations(range(1, k)):
        order = (0,) + perm
        L = loop_length(idxs, order)
        if L < best_len:
            best_len, best_order = L, order
    return best_len, best_order

def q2(selected):
    L, order = best_loop(selected)
    print("===== 问题二：站点最优运行环线 =====")
    print(f"最优环线总里程: {L:.1f} 米")
    print("顺序: " + " -> ".join(names[selected[i]] for i in order) +
          f" -> {names[selected[order[0]]]}\n")
    return L


# ========================================================================
# 3. 问题三：联合优化——枚举多个最小覆盖方案，比较环线里程，取全局最优
# ========================================================================
def q3(k_max=4, n_solutions=8):
    print("===== 问题三：站点选择与环线里程联合优化 =====")
    solutions, forbidden = [], []
    for _ in range(n_solutions):
        sol = solve_min_cover(forbidden_sets=forbidden)
        if sol is None or len(sol) > k_max:
            break
        solutions.append(sol)
        forbidden.append(list(sol))

    results = []
    for sol in solutions:
        L , _ = best_loop(sol)
        results.append((L, sol))
        print(f"  方案 {[names[i] for i in sol]}  环线里程 = {L:.1f} m")
    results.sort(key=lambda t: t[0])
    best_L, best_sol = results[0]
    print(f"\n综合最优方案: {[names[i] for i in best_sol]}  最短环线里程 = {best_L:.1f} m\n")
    return best_sol


# ========================================================================
# 4. 问题四：便利店选址（2-中位模型，按建筑人流加权）
# ========================================================================
def building_weight():
    w = np.ones(n)
    for i, nm in enumerate(names):
        if any(k in nm for k in ['教辅', '南区18-21栋', '泰塑1-8栋', '研究生公寓']):
            w[i] = 3.0
        elif any(k in nm for k in ['三食堂', '四食堂']):
            w[i] = 2.5
        elif any(k in nm for k in ['门', '广场', '田径场', '快递站', '湖']):
            w[i] = 1.0
        else:
            w[i] = 1.5
    return w

def q4(candidate_idxs):
    w = building_weight()
    best = None
    print("===== 问题四：便利店选址（2-中位模型） =====")
    for combo in combinations(candidate_idxs, 2):
        total = sum(w[j] * min(dist[j, combo[0]], dist[j, combo[1]]) for j in range(n))
        print(f"  {names[combo[0]]} + {names[combo[1]]}: 加权总距离 = {total:.1f}")
        if best is None or total < best[0]:
            best = (total, combo)
    print(f"\n推荐便利店选址站点: {names[best[1][0]]} 和 {names[best[1][1]]}\n")


if __name__ == '__main__':
    selected = q1()
    q2(selected)
    best_sol = q3()
    q2(best_sol)
    q4(best_sol)