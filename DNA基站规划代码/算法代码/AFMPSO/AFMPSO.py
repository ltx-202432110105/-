import time
import numpy as np
import pandas as pd
# from tetromino_generator import generate_special_area
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_afmpso_multiple_times(p, times):
    all_results = []
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size, num_base_stations, N, iterators = p['size'], p['num_base_stations'], p['num_particles'], p['max_iterations']
        dim = num_base_stations * 3

        # special_area = generate_special_area(size=p['size'])
        special_area = load_special_area()
        # 初始化种群个体、移动速度以及适应度值
        x = np.random.uniform(0, size, (N, dim))
        x[:, 2::3] = np.random.uniform(100, 500, (N, num_base_stations))
        v = np.zeros((N, dim))
        fitness = np.full(N, float('inf'))
        
        c1 = np.full(N, 2.0)
        c2 = np.full(N, 2.0)
        
        Xgbest = x[fitness.argmin()].copy()
        fgbest = fitness.min()
        Xpbest = x.copy()
        fpbest = fitness.copy()
        
        start_time = time.time()
        fitness_history = []
        
        for t in range(iterators):
            w = 2 / (t + 1)
            
            StudyZu = []
            rankings = np.argsort(fitness)
            for i in range(int(N/(t+1))):
                StudyZu.append(x[rankings[i]].copy())
            if len(StudyZu) == 0:
                StudyedModel = x[fitness.argmin()].copy()
            else:
                StudyedModel = np.mean(np.array(StudyZu), axis=0)

            for i in range(N):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                
                # 更新粒子速度和位置
                v[i] = w * v[i] + c1[i] * r1 * (Xpbest[i] - x[i]) + c2[i] * r2 * (Xgbest - x[i]) + r3 * (StudyedModel-x[i]) 
                x[i] += v[i]
                
                # 确保更新后的位置在取值范围内
                for j in range(dim):
                    if j % 3 < 2:  # x, y坐标限制
                        x[i][j] = max(0, min(x[i][j], size - 1))
                    else:  # 高度限制
                        x[i][j] = max(100, min(x[i][j], 500))
            
            # 计算适应度值
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            
            for i in range(N):
                base_stations = [init_base_station(x[i, j*3+2], (x[i, j*3], x[i, j*3+1])) for j in range(num_base_stations)]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[i] = objective_function(size, grid, coverage_grid, special_area)
                
                if fitness[i] < fpbest[i]:
                    fpbest[i] = fitness[i]
                    Xpbest[i] = x[i].copy()
            
            if fpbest.min() < fgbest:
                fgbest = fpbest.min()
                Xgbest = Xpbest[fpbest.argmin()].copy()

            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: {fgbest}")

            fitness_history.append(fgbest)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 根据全局最佳位置计算覆盖率和均匀性
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(Xgbest[j*3+2], (Xgbest[j*3], Xgbest[j*3+1])) for j in range(num_base_stations)]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)
        
        base_station_positions = ";".join([f"({Xgbest[j]:.2f}, {Xgbest[j+1]:.2f}, {Xgbest[j+2]:.2f})" for j in range(0, len(Xgbest), 3)])
        
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
    results = run_afmpso_multiple_times(p, times=20)

    df = pd.DataFrame(results)
    df.to_excel("AFMPSO_results.xlsx", index=False)
