import time
import numpy as np
import pandas as pd
import random
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_amsepso_multiple_times(p, times):
    all_results = []
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size, num_base_stations, N, iterators = p['size'], p['num_base_stations'], p['num_particles'], p['max_iterations']
        special_area = load_special_area()

        # 初始化种群个体、速度、适应度
        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)] for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N

        # 初始全局和个体最优
        Xpbest = [xi.copy() for xi in x]
        fpbest = fitness.copy()
        Fgbest_index = np.argmin(fpbest)
        Xgbest = Xpbest[Fgbest_index].copy()
        Fgbest = fpbest[Fgbest_index]

        Xpworst = [xi.copy() for xi in x]
        Fpworst = fitness.copy()
        Fgworst_index = np.argmax(Fpworst)
        Xgworst = Xpworst[Fgworst_index].copy()
        Fgworst = Fpworst[Fgworst_index]

        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_min, c2_max = 0.5, 2.5

        r = np.zeros(iterators + 1)
        r[0] = np.random.rand()

        start_time = time.time()
        fitness_history = []

        for t in range(iterators):
            # 动态惯性权重
            w = w_max - (w_max - w_min) * (t / iterators)

            # 动态调整加速系数
            c1 = c1_max - r[t] * t / iterators
            c2 = c2_min + r[t] * t / iterators

            # 随机扰动更新r
            u = np.random.uniform(0.9, 1.08)
            r[t+1] = u * 7.86 * r[t] + 23.31 * r[t]**2 + 28.75 * r[t]**3 + 13.28 * r[t]**4 + 0.5 * r[t]**5

            # 对种群按适应度排序
            particles = list(zip(x, v, fitness))
            particles.sort(key=lambda particle: particle[2])
            x, v, fitness = zip(*particles)
            x, v, fitness = list(x), list(v), list(fitness)

            # 更新个体最优和全局最优
            for i in range(N):
                if fitness[i] < fpbest[i]:
                    fpbest[i] = fitness[i]
                    Xpbest[i] = x[i].copy()
                else:
                    Xpworst[i] = x[i].copy()
                    Fpworst[i] = fitness[i]

            Fgbest_index = np.argmin(fpbest)
            if fpbest[Fgbest_index] < Fgbest:
                Fgbest = fpbest[Fgbest_index]
                Xgbest = Xpbest[Fgbest_index].copy()

            Fgworst_index = np.argmax(Fpworst)
            if Fpworst[Fgworst_index] > Fgworst:
                Fgworst = Fpworst[Fgworst_index]
                Xgworst = Xpworst[Fgworst_index].copy()

            # 选择两个较优个体作为 XCbest
            top_20_percent = int(N * 0.2)
            choice = random.sample(range(top_20_percent), 2)
            pos1, pos2 = choice
            if fpbest[pos1] <= fpbest[pos2]:
                XCbest = x[pos1]
                FCest = fpbest[pos1]
            else:
                XCbest = x[pos2]
                FCest = fpbest[pos2]

            XMdbest = np.mean([np.array(p) for p in Xpbest], axis=0).tolist()

            # 计算多样性指标 Dgb 和 Dgw
            Ngb, Ngw = 0, 0
            o = 0.3 * N
            for i in range(N):
                # 转换为 numpy 数组用于计算方差
                Xpbest_np = np.array(Xpbest[i])
                Xgbest_np = np.array(Xgbest)
                Xgworst_np = np.array(Xgworst)
                Xpworst_i = np.array(Xpworst[i])

                Dgb_i = np.var(Xpbest_np - Xgbest_np) / (np.var(Xgbest_np) + 1e-8)
                Dgw_i = np.var(Xpworst_i - Xgworst_np) / (np.var(Xgworst_np) + 1e-8)

                if Dgb_i < Dgw_i:
                    Ngb += 1
                else:
                    Ngw += 1

            z = Ngw - Ngb

            # 更新每个粒子的速度和位置
            for i in range(N):
                r1 = np.random.rand()
                r2 = np.random.rand()
                Spbest = Xpbest[i]
                if fpbest[i] > FCest:
                    Spbest = XCbest

                for j in range(num_base_stations):
                    for k in range(3):  # x, y, z
                        if z > o:
                            # 探索模式
                            v[i][j][k] = w * v[i][j][k] + c1 * r1 * (Xpbest[i][j][k] - x[i][j][k]) + c2 * r2 * (Xgbest[j][k] - x[i][j][k])
                        elif z < -o:
                            # 开发模式
                            v[i][j][k] = w * v[i][j][k] + c1 * r1 * (Spbest[j][k] - x[i][j][k]) + c2 * r2 * (XMdbest[j][k] - x[i][j][k])
                        else:
                            # 平衡模式
                            v[i][j][k] = w * v[i][j][k] + c1 * r1 * (Xpbest[i][j][k] - x[i][j][k]) + c2 * r2 * (XMdbest[j][k] - x[i][j][k])

                        x[i][j][k] += v[i][j][k]
                        # 边界限制
                        x[i][j][0] = max(0, min(x[i][j][0], size - 1))
                        x[i][j][1] = max(0, min(x[i][j][1], size - 1))
                        x[i][j][2] = max(100, min(x[i][j][2], 500))

            # 计算适应度
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            for h in range(N):
                base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in x[h]]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[h] = objective_function(size, grid, coverage_grid, special_area)

            # 更新个体最优与全局最优
            for i in range(N):
                if fitness[i] < fpbest[i]:
                    fpbest[i] = fitness[i]
                    Xpbest[i] = [bs.copy() for bs in x[i]]

            current_best_idx = np.argmin(fpbest)
            if fpbest[current_best_idx] < Fgbest:
                Fgbest = fpbest[current_best_idx]
                Xgbest = Xpbest[current_best_idx].copy()

            fitness_history.append(Fgbest)

        end_time = time.time()
        total_time = end_time - start_time

        # 最终评估
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in Xgbest]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)

        # 存储结果
        result = {
            'Iteration': tim + 1,
            'Fitness': Fgbest,
            'Total Coverage': total_coverage,
            'Special Area Coverage': special_coverage,
            'General Area Coverage': general_coverage,
            'Special Area Signal Uniformity': std_special,
            'General Area Signal Uniformity': std_general,
            'Time': total_time,
            'Base Station Positions': "; ".join([f"({bs[0]:.2f}, {bs[1]:.2f}, {bs[2]:.2f})" for bs in Xgbest]),
        }

        # 添加每代最优值
        for idx, val in enumerate(fitness_history):
            result[f"Iter_{idx+1}"] = val

        all_results.append(result)

    return all_results


if __name__ == "__main__":
    p = canshu()
    results = run_amsepso_multiple_times(p, times=20)

    df = pd.DataFrame(results)
    df.to_excel("AMSEPSO_results.xlsx", index=False)
    