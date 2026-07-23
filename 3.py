# -*- coding: utf-8 -*-
"""
大学校园送水站送水问题——完整求解代码
问题一：送水代理点选址（Minisum 选址模型）
问题二：单送水员配送路线（TSP，最近邻 + 2-opt）
问题三：多送水员配送（CVRP，Clarke-Wright 节约算法，考虑送水能力约束）
问题四：同时送水+收空桶（VRPSPD，沿途载重可行性校验与调整）

依赖：numpy
"""

import numpy as np

# ========================================================================
# 0. 数据
# ========================================================================
# 工作区: (X坐标/百米, Y坐标/百米, 送桶装水数量, 收空桶数量)
data = {
    'W1':  (7, 7, 14, 14),     'W2':  (20, 20, 18, 25),
    'W3':  (65.5, 91.5, 19, 19), 'W4':  (33, 55.1, 18, 27),
    'W5':  (94.1, 77.2, 24, 19), 'W6':  (62.9, 33, 16, 17),
    'W7':  (79.8, 78.5, 19, 13), 'W8':  (46, 64.2, 17, 11),
    'W9':  (77.2, 49.9, 22, 18), 'W10': (86.3, 77.2, 15, 24),
    'W11': (65.5, 22.6, 18, 12), 'W12': (7, 52.5, 23, 15),
    'W13': (31.7, 29.1, 14, 14), 'W14': (36.9, 34.3, 18, 15),
    'W15': (46, 36.9, 18, 15),   'W16': (30.4, 35.6, 17, 25),
    'W17': (65.5, 60.3, 21, 18), 'W18': (49.9, 51.2, 23, 28),
    'W19': (55.1, 44.7, 18, 22), 'W20': (51.2, 62.9, 19, 15),
}
names = list(data.keys())
n = len(names)
coords = np.array([[data[k][0], data[k][1]] for k in names])
deliver = np.array([data[k][2] for k in names])   # 送水量
pickup = np.array([data[k][3] for k in names])    # 收桶量

dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist[i, j] = np.linalg.norm(coords[i] - coords[j])


# ========================================================================
# 1. 问题一：代理点选址（Minisum / 1-中位模型）
#    在20个现有点中选1个点 i*，使 sum_j w_j * dist(i,j) 最小
#    权重 w_j 取该点日均业务总量（送水量+收桶量）
# ========================================================================
def solve_location():
    weight = deliver + pickup
    total_weighted_dist = np.array([np.sum(weight * dist[i, :]) for i in range(n)])
    best = np.argmin(total_weighted_dist)
    print("===== 问题一：代理点选址 =====")
    for i in np.argsort(total_weighted_dist)[:3]:
        print(f"  {names[i]}: 加权距离和 = {total_weighted_dist[i]:.2f}")
    print(f"  推荐代理点: {names[best]}\n")
    return best


# ========================================================================
# 2. 问题二：单人 TSP（最近邻构造 + 2-opt 局部优化）
# ========================================================================
def route_length(route):
    return sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1))

def nearest_neighbor(start, nodes):
    unvisited = set(nodes) - {start}
    route, cur = [start], start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur, j])
        route.append(nxt); unvisited.remove(nxt); cur = nxt
    return route

def two_opt(route):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b, c, d = best[i-1], best[i], best[j], best[j+1]
                delta = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                if delta < -1e-9:
                    best[i:j+1] = best[i:j+1][::-1]
                    improved = True
    return best

def solve_tsp(depot):
    route = nearest_neighbor(depot, list(range(n)))
    route.append(depot)
    route = two_opt(route)
    print("===== 问题二：单送水员配送路线 =====")
    print(f"  路线总里程: {route_length(route):.2f} 百米")
    print("  " + " -> ".join(names[i] for i in route) + "\n")
    return route


# ========================================================================
# 3. 问题三：多送水员 CVRP（Clarke-Wright 节约算法，容量=送水能力）
# ========================================================================
def savings_cvrp(depot, customers, demand, capacity):
    routes = {c: [depot, c, depot] for c in customers}
    route_of = {c: c for c in customers}
    load = {c: demand[c] for c in customers}

    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = dist[depot, i] + dist[depot, j] - dist[i, j]
                savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    for s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        route_i, route_j = routes[ri], routes[rj]
        if route_i[-2] == i and route_j[1] == j:
            merged = route_i[:-1] + route_j[1:]
        elif route_j[-2] == j and route_i[1] == i:
            merged = route_j[:-1] + route_i[1:]
        else:
            continue
        new_load = load[ri] + load[rj]
        if new_load > capacity:
            continue
        key = ri
        routes[key] = merged
        load[key] = new_load
        for c in merged:
            if c != depot:
                route_of[c] = key
        del routes[rj]; del load[rj]

    return [two_opt(r) for r in routes.values()]

def solve_cvrp(depot, capacity):
    customers = [i for i in range(n) if i != depot]
    demand = {i: deliver[i] for i in range(n)}
    total_demand = sum(deliver[i] for i in customers)
    min_vehicles = int(np.ceil(total_demand / capacity))
    routes = savings_cvrp(depot, customers, demand, capacity)
    total_len = sum(route_length(r) for r in routes)
    print(f"===== 问题三：多送水员配送（送水能力 Q = {capacity}） =====")
    print(f"  理论最少车辆数: {min_vehicles}  实际使用车辆数: {len(routes)}")
    print(f"  总配送里程: {total_len:.2f} 百米")
    for k, r in enumerate(routes, 1):
        load = sum(deliver[c] for c in r if c != depot)
        print(f"  路线{k}(载重 {load}/{capacity}): " + " -> ".join(names[i] for i in r))
    print()
    return routes


# ========================================================================
# 4. 问题四：同时送水+收空桶（VRPSPD）
#    出发载重 = 该路线总送水量；沿途每站 load = load - deliver_i + pickup_i
#    需保证任意时刻 0 <= load <= capacity，否则该方案不可行，需调整路线
# ========================================================================
def simulate_load(route, depot, capacity):
    total_deliver = sum(deliver[c] for c in route if c != depot)
    load = total_deliver
    trace = [load]
    ok = load <= capacity
    for c in route[1:-1]:
        load = load - deliver[c] + pickup[c]
        trace.append(load)
        if load > capacity or load < 0:
            ok = False
    return trace, ok

def check_pickup_delivery(routes, depot, capacity):
    print(f"===== 问题四：送收同时进行时的载重校验（Q = {capacity}） =====")
    all_ok = True
    for k, r in enumerate(routes, 1):
        trace, ok = simulate_load(r, depot, capacity)
        status = "可行" if ok else "不可行，需重新分配/调整路线"
        print(f"  路线{k} 沿途载重: {trace} 最大值={max(trace)} -> {status}")
        all_ok = all_ok and ok
    print(f"  总体结论: {'现有路线满足送收能力约束，无需调整' if all_ok else '需调整路线（如在路线间调换工作区或重新聚类）'}\n")
    return all_ok


# ========================================================================
# 主程序
# ========================================================================
if __name__ == '__main__':
    depot = solve_location()
    solve_tsp(depot)

    for Q in [200, 195, 190]:
        routes = solve_cvrp(depot, Q)
        check_pickup_delivery(routes, depot, Q)