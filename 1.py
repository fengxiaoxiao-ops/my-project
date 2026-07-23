# -*- coding: utf-8 -*-
"""
武汉工程大学校园临时集中停车场所优化布局——完整求解代码

问题背景：为大型活动临时开放的集中停车场地划定车位，比较垂直式(90°)、
斜列式(30°/45°/60°)、平行式三类停车位布局，并分别给出"一对出入口"和
"两对出入口"两种情况下的车位划分方案与定量优化指标。

模型思路：
1. 场地假设为矩形（依据见下方注释），扣除周边安全退线得到可用矩形。
2. 每类车位形成"双侧停车带模块"：模块厚度 = 2×车位进深 + 通道宽度；
   模块能沿场地长度方向排下的车位数 = floor(长度 / 车位沿边宽度)。
3. 单一对出入口：所有通道必须双向通行(通道更宽)，且入口端需预留回车缓冲区；
   两对出入口：可组织单向环形流线，斜列/平行式的通道可显著收窄，90°因转弯
   几何限制通道宽度不能显著减小。
4. 用无界背包搜索纯数学最优的车位类型组合(帮助理解上限)，
   再给出"综合应用三类车位"的推荐工程方案(便于管理、满足题目要求)，
   并计算面积利用率、车均占地、平均行驶距离等定量指标。

依赖：numpy
"""
import numpy as np
from collections import Counter


# ========================================================================
# 0. 场地假设
# 依据：题目图2卫星示意图显示该地块位于东校门-凤鸣山-芳郁大道-自强大道
# 所围空地，无实测边界数据。参照示意图比例与相邻标准田径场(约170m×90m)
# 的相对尺度估算，取一个较为保守、便于计算的矩形近似：
#   长度方向(沿自强大道) L0 = 150m，宽度方向(沿芳郁大道) W0 = 80m
#   面积 S0 = 12000 m^2
# 若能获得实测边界坐标，只需替换 L0, W0（或改造成多边形分区）即可。
# ========================================================================
L0, W0 = 150.0, 80.0
SETBACK = 1.0          # 周边安全退线(m)，参考消防/安全通道要求
L = L0 - 2 * SETBACK
W = W0 - 2 * SETBACK
SITE_AREA = L0 * W0
TURN_APRON = 12.0       # 单入口时端部回车缓冲区深度(m)

# ========================================================================
# 1. 三类车位标准尺寸（参考《车库建筑设计规范》JGJ100-2015、
#    《城市道路路内停车泊位设置规范》对小型车的常用工程取值）
# ========================================================================
PARK_TYPES = {
    '垂直式90°': dict(stall_width=2.5, stall_depth=5.3, aisle_2way=5.5, aisle_1way=5.5),
    '斜列式60°': dict(stall_width=2.9, stall_depth=5.7, aisle_2way=4.5, aisle_1way=3.5),
    '斜列式45°': dict(stall_width=3.5, stall_depth=5.5, aisle_2way=3.8, aisle_1way=3.5),
    '斜列式30°': dict(stall_width=5.0, stall_depth=4.8, aisle_2way=3.5, aisle_1way=3.0),
    '平行式':    dict(stall_width=6.0, stall_depth=2.5, aisle_2way=3.5, aisle_1way=3.5),
}


def module_info(ptype, n_entrances):
    """返回该类型双侧停车带模块的厚度(m)和单条车带可容纳车位数"""
    p = PARK_TYPES[ptype]
    # 90°转弯几何限制，单向/双向通道宽度取同一值；其余类型双入口时可用较窄单向通道
    aisle = p['aisle_2way'] if (n_entrances == 1 or ptype == '垂直式90°') else p['aisle_1way']
    depth = 2 * p['stall_depth'] + aisle
    stalls_per_side = int(L // p['stall_width'])
    return depth, 2 * stalls_per_side


# ========================================================================
# 2. 单一类型纯布局（用于横向对比不同车位类型/出入口数的效率）
# ========================================================================
def pure_layout(ptype, n_entrances):
    depth, stalls = module_info(ptype, n_entrances)
    usable_W = W - (TURN_APRON if n_entrances == 1 else 0)
    n_strips = int(usable_W // depth)
    total = n_strips * stalls
    return dict(type=ptype, n_entrances=n_entrances, n_strips=n_strips,
                total=total, area_per_stall=SITE_AREA / total if total else np.inf)


# ========================================================================
# 3. 数学最优组合（无界背包）：不限制车位类型种类，纯以总车位数最大化为目标
# ========================================================================
def knapsack_mix(n_entrances):
    usable_W = W - (TURN_APRON if n_entrances == 1 else 0)
    step = 0.1
    cap = int(round(usable_W / step))
    items = [(ptype, int(round(module_info(ptype, n_entrances)[0] / step)),
              module_info(ptype, n_entrances)[1]) for ptype in PARK_TYPES]

    dp = np.zeros(cap + 1, dtype=int)
    choice = [[] for _ in range(cap + 1)]
    for c in range(1, cap + 1):
        best, best_choice = dp[c - 1], choice[c - 1] + [None]
        for ptype, w, stalls in items:
            if w <= c and dp[c - w] + stalls > best:
                best, best_choice = dp[c - w] + stalls, choice[c - w] + [ptype]
        dp[c], choice[c] = best, best_choice
    counts = Counter(t for t in choice[cap] if t is not None)
    return dp[cap], counts


# ========================================================================
# 4. 推荐工程方案：按题目要求"综合应用三类车位"，兼顾管理便利性与安全
#    （在数学最优解附近微调，纳入平行式/斜列式用于场地边界与出入口过渡区）
# ========================================================================
def build_design(n_entrances, plan):
    usable_W = W - (TURN_APRON if n_entrances == 1 else 0)
    used_depth, total_stalls, rows = 0.0, 0, []
    for ptype, k in plan:
        depth, stalls = module_info(ptype, n_entrances)
        used_depth += depth * k
        total_stalls += stalls * k
        rows.append((ptype, k, depth, stalls))
    leftover = usable_W - used_depth
    area_used = used_depth * L
    return dict(rows=rows, used_depth=used_depth, leftover=leftover,
                total=total_stalls, area_used=area_used,
                utilization=area_used / SITE_AREA)


def avg_travel_distance(n_entrances, n_strips_total, module_depth_avg):
    long_dist = L if n_entrances == 1 else L / 2   # 单入口需往返；双入口单向穿越
    lateral_dist = n_strips_total / 2 * module_depth_avg
    return long_dist + lateral_dist


# ========================================================================
# 主程序
# ========================================================================
if __name__ == '__main__':
    print(f"场地假设: {L0}m x {W0}m, 面积 {SITE_AREA:.0f} m^2\n")

    print("===== 各类型纯布局对比 =====")
    for ptype in PARK_TYPES:
        for ne in [1, 2]:
            r = pure_layout(ptype, ne)
            print(f"  {ptype:<8s} 出入口{ne}对: 车带{r['n_strips']}条, "
                  f"总车位{r['total']:4d}, 车均占地{r['area_per_stall']:.1f} m2")

    print("\n===== 数学最优组合(无界背包，不限类型种类) =====")
    for ne in [1, 2]:
        total, counts = knapsack_mix(ne)
        combo = ", ".join(f"{t}x{c}" for t, c in counts.items())
        print(f"  出入口{ne}对: 最优组合[{combo}] 总车位 {total}")

    print("\n===== 推荐工程方案(综合三类车位) =====")
    plan1 = [('垂直式90°', 3), ('斜列式45°', 1)]
    d1 = build_design(1, plan1)
    print(f"  [单入口] " + "; ".join(f"{t}x{k}条({s}位/条)" for t, k, dep, s in d1['rows']))
    print(f"    总车位 {d1['total']}, 面积利用率 {d1['utilization']*100:.1f}%, "
          f"车均占地 {SITE_AREA/d1['total']:.1f} m2")

    plan2 = [('垂直式90°', 2), ('斜列式60°', 2), ('平行式', 1)]
    d2 = build_design(2, plan2)
    print(f"  [双入口] " + "; ".join(f"{t}x{k}条({s}位/条)" for t, k, dep, s in d2['rows']))
    print(f"    总车位 {d2['total']}, 面积利用率 {d2['utilization']*100:.1f}%, "
          f"车均占地 {SITE_AREA/d2['total']:.1f} m2")

    t1 = avg_travel_distance(1, 4, 16.1)
    t2 = avg_travel_distance(2, 5, 15.3)
    print(f"\n  单入口方案 平均行驶距离估算: {t1:.0f} m")
    print(f"  双入口方案 平均行驶距离估算: {t2:.0f} m "
          f"(缩短约 {(1 - t2/t1)*100:.0f}%)")