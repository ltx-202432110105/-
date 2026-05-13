import time
import numpy as np
import pandas as pd
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_fhpso_within_time(p, max_runtime_seconds, times):
    all_results = []
    fixed_time_columns = [f"time_{i}" for i in range(100, max_runtime_seconds + 1, 100)]

    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")

        size = p['size']
        num_base_stations = p['num_base_stations']
        N = p['num_particles']
        special_area = load_special_area()

        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)] for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N

        # 参数设置
        w = 0.9
        c1 = 2
        c2 = 2

        # 分组设置
        ZuLevel = [[0, int(N*0.1)], [int(N*0.1), int(N*0.3)], [int(N*0.3), int(N*0.6)], [int(N*0.6), N]]

        # 全局最优初始化
        Xgbest = x[fitness.index(min(fitness))].copy()
        fgbest = min(fitness)
        Xpbest = [xi.copy() for xi in x]
        fpbest = fitness.copy()

        start_time = time.time()
        result_fitness_history = {}
        next_target_time = 100  # 下一个记录时间点 (秒)

        while time.time() - start_time < max_runtime_seconds:
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)

            # 计算当前所有粒子的适应度
            for h in range(N):
                base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in x[h]]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[h] = objective_function(size, grid, coverage_grid, special_area)

            # 排序粒子：按照适应度升序排列（越小越好）
            idx = np.argsort(fitness)
            x_sorted = [x[i] for i in idx]
            v_sorted = [v[i] for i in idx]
            fitness_sorted = [fitness[i] for i in idx]
            Xpbest_sorted = [Xpbest[i] for i in idx]
            fpbest_sorted = [fpbest[i] for i in idx]

            # 替换回原数组
            x = x_sorted
            v = v_sorted
            fitness = fitness_sorted
            Xpbest = Xpbest_sorted
            fpbest = fpbest_sorted

            # 更新全局最优
            current_fmin = min(fpbest)
            if current_fmin < fgbest:
                fgbest = current_fmin
                Xgbest = Xpbest[fpbest.index(fgbest)].copy()

            # 检查是否到达下一个记录时间点
            current_time = time.time()
            elapsed_time = int(current_time - start_time)

            while next_target_time <= max_runtime_seconds and elapsed_time >= next_target_time:
                result_fitness_history[f"time_{next_target_time}"] = fgbest
                next_target_time += 100

            # 分组信息记录
            ZuInfo = []

            for num in range(4):
                start, end = ZuLevel[num]
                group_fitness = fitness[start:end]
                group_x = x[start:end]

                Em = np.mean([np.array(bs) for ind in range(start, end) for bs in x[ind]], axis=0)
                fm = np.mean(group_fitness)

                indices = [i for i in range(start, end) if fitness[i - start] <= fm]
                Ehm = np.mean([np.array(bs) for i in indices for bs in x[i]], axis=0) if len(indices) > 0 else Em

                ZuInfo.append([Em, start + indices[-1] if len(indices) > 0 else end - 1, Ehm])

            # 更新粒子速度和位置
            for i in range(N):
                r1 = np.random.rand()
                r2 = np.random.rand()

                for j in range(num_base_stations):
                    for k in range(3):  # x, y, z
                        for num in range(4):
                            start, end = ZuLevel[num]
                            if start <= i < end:
                                if i <= ZuInfo[num][1]:  # 高能粒子
                                    a = np.random.rand()
                                    pre_idx = num - 1 if num > 0 else num
                                    v[i][j][k] = w * v[i][j][k] + \
                                                 c1 * r1 * (Xpbest[i][j][k] - x[i][j][k]) + \
                                                 c2 * r2 * (Xgbest[j][k] - x[i][j][k]) + \
                                                 a * (ZuInfo[pre_idx][0][k] - x[i][j][k])
                                else:  # 低能粒子
                                    v[i][j][k] = w * v[i][j][k] + \
                                                 c1 * r1 * (Xpbest[i][j][k] - x[i][j][k]) + \
                                                 r2 * (ZuInfo[num][2][k] - x[i][j][k])
                                break
                        x[i][j][k] += v[i][j][k]

                    # 边界处理
                    x[i][j][0] = max(0, min(x[i][j][0], size - 1))
                    x[i][j][1] = max(0, min(x[i][j][1], size - 1))
                    x[i][j][2] = max(100, min(x[i][j][2], 500))

            # 更新个体历史最优
            for h in range(N):
                if fitness[h] < fpbest[h]:
                    fpbest[h] = fitness[h]
                    Xpbest[h] = [bs.copy() for bs in x[h]]

        # 填充未记录的时间点为最终 fgbest
        for col in fixed_time_columns:
            if col not in result_fitness_history:
                result_fitness_history[col] = fgbest

        end_time = time.time()
        total_time = end_time - start_time

        # 最终评估
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in Xgbest]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)

        base_station_positions = ";".join([f"({bs[0]:.2f}, {bs[1]:.2f}, {bs[2]:.2f})" for bs in Xgbest])

        result = {
            'Iteration': tim + 1,
            'Fitness': fgbest,
            'Total Coverage': total_coverage,
            'Special Area Coverage': special_coverage,
            'General Area Coverage': general_coverage,
            'Special Area Signal Uniformity': std_special,
            'General Area Signal Uniformity': std_general,
            'Time': total_time,
            'Base Station Positions': base_station_positions
        }

        # 添加固定时间间隔的适应度记录
        result.update(result_fitness_history)

        all_results.append(result)

    return all_results

if __name__ == "__main__":
    p = canshu()
    max_runtime_seconds = 1700
    results = run_fhpso_within_time(p, max_runtime_seconds=max_runtime_seconds, times=20)

    df = pd.DataFrame(results)
    df.to_excel("FHPSO_t.xlsx", index=False)
