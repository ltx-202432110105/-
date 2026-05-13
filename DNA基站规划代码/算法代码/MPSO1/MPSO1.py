import time
import numpy as np
import pandas as pd
import random
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_mpso_multiple_times(p, times):
    all_results = []
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size, num_base_stations, N, iterators = p['size'], p['num_base_stations'], p['num_particles'], p['max_iterations']
        special_area = load_special_area()

        # 初始化种群个体、移动速度以及适应度值
        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)] for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N
        
        w_max = 0.9
        w_min = 0.4
        c1 = 2
        c2 = 2

        # 新增变量
        Xpbest = [xi.copy() for xi in x]
        Fpbest = fitness.copy()
        Fgbest = min(Fpbest)
        Xgbest = Xpbest[Fpbest.index(Fgbest)].copy()

        # 混沌映射控制惯性权重
        r = np.zeros(iterators)
        r[0] = np.random.rand()
        while r[0] in [0, 0.25, 0.5, 0.75, 1]:
            r[0] = np.random.rand()

        fitness_history = []

        start_time = time.time()

        for t in range(iterators):
            if t < iterators - 1:
                r[t+1] = 4 * r[t] * (1 - r[t])
            w = r[t] * w_min + (w_max - w_min) * (t+1) / iterators

            # 随机选两个个体比较选出较优XCbest
            choice = random.sample(range(N), 2)
            pos1, pos2 = choice
            if Fpbest[pos1] <= Fpbest[pos2]:
                XCbest_idx = pos1
                FCest = Fpbest[pos1]
            else:
                XCbest_idx = pos2
                FCest = Fpbest[pos2]
            XCbest = Xpbest[XCbest_idx]

            # 种群最优平均XMbest
            XMbest = [np.mean([Xpbest[i][j][k] for i in range(N)], axis=0) 
                      for j in range(num_base_stations) for k in range(3)]
            XMbest = [[XMbest[j*3 + k] for k in range(3)] for j in range(num_base_stations)]

            # 平均适应度
            avg_fitness = np.mean(Fpbest)

            # 粒子更新
            for i in range(N):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                for j in range(num_base_stations):
                    for k in range(3):  # x, y, z
                        # 根据适应度选择Spbest
                        Spbest = Xpbest[i][j] if Fpbest[i] <= FCest else XCbest[j]
                        
                        # 更新速度
                        v[i][j][k] =w*v[i][j][k]+c1*r1*(Spbest[k]-x[i][j][k])+c2*r2*(Xgbest[j][k]-x[i][j][k])

                        # 社会现象模拟（概率扰动）
                        p_prob = np.exp(Fpbest[i]) / np.exp(avg_fitness)
                        if p_prob > np.random.rand():
                            v[i][j][k] = w * v[i][j][k] + (1 - w) * v[i][j][k] + Xgbest[j][k]
                        else:
                            x[i][j][k] += v[i][j][k]

                        # 边界限制
                        if k == 0 or k == 1:  # x,y坐标
                            x[i][j][k] = max(0, min(x[i][j][k], size - 1))
                        elif k == 2:  # 高度
                            x[i][j][k] = max(100, min(x[i][j][k], 500))

            # 计算适应度
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            for h in range(N):
                base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in x[h]]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[h] = objective_function(size, grid, coverage_grid, special_area)

                if fitness[h] < Fpbest[h]:
                    Fpbest[h] = fitness[h]
                    Xpbest[h] = [bs.copy() for bs in x[h]]

            # 更新全局最优
            if min(Fpbest) < Fgbest:
                Fgbest = min(Fpbest)
                Xgbest = Xpbest[Fpbest.index(Fgbest)].copy()

            # 替换最差个体
            worst_idx = Fpbest.index(max(Fpbest))
            b = np.random.rand()
            choice = random.sample(range(N), 2)
            pos1, pos2 = choice
            Nbest = [
                [x[pos1][j][k] + b * (x[pos1][j][k] - x[pos2][j][k]) for k in range(3)]
                for j in range(num_base_stations)
            ]
            x[worst_idx] = Nbest
            Fpbest[worst_idx] = float('inf')  # 强制重新计算

            # 记录
            fitness_history.append(Fgbest)
            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: {Fgbest}")

        end_time = time.time()
        total_time = end_time - start_time

        # 结果统计
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in Xgbest]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)

        base_station_positions = "; ".join([f"({bs[0]:.2f}, {bs[1]:.2f}, {bs[2]:.2f})" for bs in Xgbest])

        result = {
            'Iteration': tim + 1,
            'Fitness': Fgbest,
            'Total Coverage': total_coverage,
            'Special Area Coverage': special_coverage,
            'General Area Coverage': general_coverage,
            'Special Area Signal Uniformity': std_special,
            'General Area Signal Uniformity': std_general,
            'Time': total_time,
            'Base Station Positions': base_station_positions
        }

        for t, value in enumerate(fitness_history):
            result[f"time {t + 1}"] = value

        all_results.append(result)

    return all_results


if __name__ == "__main__":
    p = canshu()
    results = run_mpso_multiple_times(p, times=20)

    df = pd.DataFrame(results)
    df.to_excel("MPSO_results.xlsx", index=False)
