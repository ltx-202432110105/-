import time
import numpy as np
import pandas as pd
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def metropolis_criterion(new_fitness, old_fitness, temperature):
    """模拟退火算法的Metropolis准则"""
    if new_fitness < old_fitness:
        return True
    else:
        p = np.exp(-(new_fitness - old_fitness) / temperature)
        return p > np.random.rand()

def dimensional_learning(particle, global_best, m_threshold, current_iter, max_iter):
    """维度学习策略"""
    if current_iter > m_threshold:
        for j in range(len(particle)):
            dim = np.random.randint(0, 3)  # 随机选择x/y/z维度
            particle[j][dim] = global_best[j][dim]
    return particle

def run_sdpso_within_time(p, max_runtime_seconds, times):
    all_results = []
    fixed_time_columns = [f"time_{i}" for i in range(100, max_runtime_seconds + 1, 100)]

    for tim in range(times):
        print(f"Running SDPSO iteration {tim + 1}/{times}")

        size = p['size']
        num_base_stations = p['num_base_stations']
        N = p['num_particles']

        special_area = load_special_area()

        # 初始化种群
        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)]
              for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N

        # 初始温度和DLS阈值
        initial_temp = 1000
        m_ratio = 0.3
        m_threshold = int(m_ratio * max_runtime_seconds)  # 根据总运行时间估算DLS开始时间点

        # 全局和个体最优初始化
        Xgbest = x[fitness.index(min(fitness))].copy()
        fgbest = min(fitness)
        Xpbest = [xi.copy() for xi in x]
        fpbest = fitness.copy()

        # 动态参数初始化
        w_max, w_min = 0.9, 0.4
        c1_start, c1_end = 2.0, 0.4
        c2_start, c2_end = 0.4, 2.0

        start_time = time.time()
        result_fitness_history = {}
        next_target_time = 100  # 下一个记录时间点 (秒)

        t = 0  # 迭代计数器
        while time.time() - start_time < max_runtime_seconds:
            current_time = time.time()
            elapsed_time = int(current_time - start_time)

            # 更新动态参数
            progress_ratio = elapsed_time / max_runtime_seconds
            w = w_max - (w_max - w_min) * progress_ratio
            temperature = initial_temp * (1 - progress_ratio)
            c1 = c1_start - (c1_start - c1_end) * progress_ratio
            c2 = c2_start + (c2_end - c2_start) * progress_ratio

            # 粒子更新
            for i in range(N):
                r1, r2 = np.random.rand(), np.random.rand()

                # 1. 标准PSO速度更新
                for j in range(num_base_stations):
                    for k in range(3):
                        v[i][j][k] = w * v[i][j][k] + \
                                    c1 * r1 * (Xpbest[i][j][k] - x[i][j][k]) + \
                                    c2 * r2 * (Xgbest[j][k] - x[i][j][k])
                        x[i][j][k] += v[i][j][k]

                # 2. DLS维度学习
                x[i] = dimensional_learning(x[i], Xgbest, m_threshold, t, max_runtime_seconds)

                # 边界处理
                for j in range(num_base_stations):
                    x[i][j][0] = np.clip(x[i][j][0], 0, size - 1)
                    x[i][j][1] = np.clip(x[i][j][1], 0, size - 1)
                    x[i][j][2] = np.clip(x[i][j][2], 100, 500)

            # 计算适应度
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)

            for h in range(N):
                base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in x[h]]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[h] = objective_function(size, grid, coverage_grid, special_area)

                # 更新个体最优
                if fitness[h] < fpbest[h]:
                    fpbest[h] = fitness[h]
                    Xpbest[h] = [bs.copy() for bs in x[h]]

            # 3. SA全局最优更新
            current_best_idx = np.argmin(fpbest)
            if metropolis_criterion(fpbest[current_best_idx], fgbest, temperature):
                fgbest = fpbest[current_best_idx]
                Xgbest = [bs.copy() for bs in Xpbest[current_best_idx]]

            # 检查是否到达下一个记录时间点
            while elapsed_time >= next_target_time and next_target_time <= max_runtime_seconds:
                result_fitness_history[f"time_{next_target_time}"] = fgbest
                next_target_time += 100

            t += 1

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
        total_coverage, special_coverage, general_coverage, std_special, std_general = \
            calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)

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
    results = run_sdpso_within_time(p, max_runtime_seconds=max_runtime_seconds, times=20)

    df = pd.DataFrame(results)
    df.to_excel("SDPSO_t.xlsx", index=False)
