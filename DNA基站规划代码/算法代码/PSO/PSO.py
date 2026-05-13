import time
import numpy as np
import pandas as pd
# from tetromino_generator import generate_special_area
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_pso_multiple_times(p, times):
    all_results = []
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size, num_base_stations, N, iterators = p['size'], p['num_base_stations'], p['num_particles'], p['max_iterations']

        # special_area = generate_special_area(size=p['size'])
        special_area = load_special_area()
        
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
        fitness_history = []
        
        for t in range(iterators):
            w = w_max - (w_max - w_min) * (t / iterators)
            
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

            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: {fgbest}")
            
            # 记录当前迭代的目标函数值
            fitness_history.append(fgbest)
        
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
        
        result[''] = None
        for t, value in enumerate(fitness_history):
            result[f"time {t + 1}"] = value
        
        all_results.append(result)
    
    return all_results

if __name__ == "__main__":
    p = canshu()
    results = run_pso_multiple_times(p, times=20)

    df = pd.DataFrame(results)
    df.to_excel("PSO_results.xlsx", index=False)
