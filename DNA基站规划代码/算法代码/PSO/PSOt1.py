import time
import numpy as np
import pandas as pd
from tetromino_generator import generate_special_area
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_pso_within_time(p, max_runtime_seconds, times):
    all_results = []
    
    # 固定的时间间隔列名
    fixed_time_columns = [f"time_{i}" for i in range(100, max_runtime_seconds + 1, 100)]
    
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size, num_base_stations, N = p['size'], p['num_base_stations'], p['num_particles']
        
        special_area = generate_special_area(size=p['size'])
        # special_area = load_special_area()
        
        # 初始化种群个体、移动速度以及适应度值
        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)] for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N
        
        w_max = 0.9
        w_min = 0.4
        c1 = 1.6
        c2 = 1.8

        Xgbest = x[fitness.index(min(fitness))].copy()
        fgbest = min(fitness)
        Xpbest = [xi.copy() for xi in x]
        fpbest = fitness.copy()
        
        start_time = time.time()
        fitness_history = {}  # 存储每次记录的目标函数值
        next_target_time = 100  # 下一个目标时间点
        t = 0
        
        while time.time() - start_time < max_runtime_seconds:
            # 动态调整惯性权重 w
            elapsed_time = time.time() - start_time
            w = w_max - (w_max - w_min) * (elapsed_time / max_runtime_seconds)
            
            for i in range(N):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # 更新粒子速度和位置
                for j in range(num_base_stations):
                    for k in range(3):  # x, y, z
                        v[i][j][k] = w * v[i][j][k] + c1 * r1 * (Xpbest[i][j][k] - x[i][j][k]) + c2 * r2 * (Xgbest[j][k] - x[i][j][k])
                        x[i][j][k] += v[i][j][k]
                    
                    # 确保更新后的位置在取值范围内
                    x[i][j][0] = max(0, min(x[i][j][0], size - 1))  # x坐标限制
                    x[i][j][1] = max(0, min(x[i][j][1], size - 1))  # y坐标限制
                    x[i][j][2] = max(100, min(x[i][j][2], 500))     # 高度限制
            
            # 计算适应度值
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            
            for h in range(N):
                base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in x[h]]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[h] = objective_function(size, grid, coverage_grid, special_area)
                
                if fitness[h] < fpbest[h]:
                    fpbest[h] = fitness[h]
                    Xpbest[h] = [bs.copy() for bs in x[h]]
            
            if min(fpbest) < fgbest:
                fgbest = min(fpbest)
                Xgbest = Xpbest[fpbest.index(fgbest)].copy()

            # 检查是否达到下一个目标时间点
            current_time = time.time()
            elapsed_time = int(current_time - start_time)
            while elapsed_time >= next_target_time and next_target_time <= max_runtime_seconds:
                fitness_history[f"time_{next_target_time}"] = fgbest
                next_target_time += 100

            t += 1
        
        # 填充未达到的时间点
        for col in fixed_time_columns:
            if col not in fitness_history:
                fitness_history[col] = fgbest  # 使用最新的 fgbest 填充

        end_time = time.time()
        total_time = end_time - start_time
        
        # 根据全局最佳位置计算覆盖率和均匀性
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in Xgbest]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)
        
        base_station_positions = ";".join([f"({bs[0]:.2f}, {bs[1]:.2f}, {bs[2]:.2f})" for bs in Xgbest])
        
        # 构建结果字典
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
        result.update(fitness_history)
        
        all_results.append(result)
    
    return all_results

if __name__ == "__main__":
    p = canshu()
    max_runtime_seconds = 1700
    results = run_pso_within_time(p, max_runtime_seconds=max_runtime_seconds, times=20)

    df = pd.DataFrame(results)
    df.to_excel("PSO_t1.xlsx", index=False)
    