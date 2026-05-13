import time
import numpy as np
import pandas as pd
import random
from tetromino_generator import generate_special_area
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_amsepso_within_time(p, max_runtime_seconds, times):
    all_results = []
    fixed_time_columns = [f"time_{i}" for i in range(100, max_runtime_seconds + 1, 100)]

    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")

        size, num_base_stations, N = p['size'], p['num_base_stations'], p['num_particles']
        special_area = generate_special_area(size=p['size'])

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

        r = np.zeros((10000, ))  # 预分配扰动序列
        r[0] = np.random.rand()

        start_time = time.time()
        next_target_time = 100

        # 用于记录每个固定时间点的适应度
        fitness_history = {}

        iteration_counter = 0  # 仅用于调试或扰动更新

        while time.time() - start_time < max_runtime_seconds:
            current_time = time.time()
            elapsed_time = int(current_time - start_time)

            # 动态惯性权重
            w = w_max - (w_max - w_min) * (elapsed_time / max_runtime_seconds)

            # 动态调整加速系数
            c1 = c1_max - r[iteration_counter % len(r)] * iteration_counter / max_runtime_seconds
            c2 = c2_min + r[iteration_counter % len(r)] * iteration_counter / max_runtime_seconds

            # 更新随机扰动r
            u = np.random.uniform(0.9, 1.08)
            r[iteration_counter + 1] = u * (
                7.86 * r[iteration_counter] +
                23.31 * r[iteration_counter]**2 +
                28.75 * r[iteration_counter]**3 +
                13.28 * r[iteration_counter]**4 +
                0.5 * r[iteration_counter]**5
            )

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

                        v[i][j][k] = np.clip(v[i][j][k], -100, 100)
                        x[i][j][k] += v[i][j][k]

                        # 边界限制
                        x[i][j][0] = max(0, min(x[i][j][0], size - 1))
                        x[i][j][1] = max(0, min(x[i][j][1], size - 1))
                        x[i][j][2] = max(100, min(x[i][j][2], 500))

            # 评估适应度
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

            # 检查并记录当前时间点的 Fgbest
            while elapsed_time >= next_target_time and next_target_time <= max_runtime_seconds:
                fitness_history[f"time_{next_target_time}"] = Fgbest
                next_target_time += 100

            iteration_counter += 1

        # 补全未记录的时间点，使用最终 fgbest 填充
        for col in fixed_time_columns:
            if col not in fitness_history:
                fitness_history[col] = Fgbest

        end_time = time.time()
        total_time = end_time - start_time

        # 最终评估
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in Xgbest]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)

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

        result.update(fitness_history)
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    p = canshu()
    max_runtime_seconds = 1700
    results = run_amsepso_within_time(p, max_runtime_seconds=max_runtime_seconds, times=20)

    df = pd.DataFrame(results)
    df.to_excel("AMSEPSO_t1.xlsx", index=False)
