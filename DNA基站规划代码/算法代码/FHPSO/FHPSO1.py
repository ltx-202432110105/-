import time
import numpy as np
import pandas as pd
from tetromino_generator import generate_special_area
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_fhpso_multiple_times(p, times):
    all_results = []
    
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size = p['size']
        num_base_stations = p['num_base_stations']
        N = p['num_particles']
        iterators = p['max_iterations']

        special_area = generate_special_area(size=p['size'])
        
        # 初始化种群：每个粒子是一个基站列表 [x, y, z]
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
        fitness_history = []

        for t in range(iterators):
            # 计算当前所有粒子的适应度
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            
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

            # 更新全局最优
            current_fmin = min(fpbest)
            if current_fmin < fgbest:
                fgbest = current_fmin
                Xgbest = Xpbest[fpbest.index(fgbest)].copy()

            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: {fgbest}")

            fitness_history.append(fgbest)

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

        for t, value in enumerate(fitness_history):
            result[f"time {t + 1}"] = value

        all_results.append(result)

    return all_results


if __name__ == "__main__":
    p = canshu()
    results = run_fhpso_multiple_times(p, times=20)

    df = pd.DataFrame(results)
    df.to_excel("FHPSO_results1.xlsx", index=False)
    