import time
import numpy as np
import pandas as pd
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, visualize, final_pic, load_special_area

def run_ahdpso_multiple_times(p, times):
    all_results = []
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        size, num_base_stations, N, iterators = p['size'], p['num_base_stations'], p['num_particles'], p['max_iterations']
        num_base_stations = 35
        dim = num_base_stations * 3
        special_area = load_special_area()
        
        x = np.random.uniform(0, size, (N, dim))
        x[:, 2::3] = np.random.uniform(100, 500, (N, num_base_stations))
        v = np.zeros((N, dim))
        fitness = np.full(N, float('inf'))
        
        Xgbest = x[fitness.argmin()].copy()
        fgbest = fitness.min()
        Xpbest = x.copy()
        fpbest = fitness.copy()
        
        start_time = time.time()
        fitness_history = []
        
        for t in range(iterators):
            w = params['w'][t]
            c1 = params['c1'][t]
            c2 = params['c2'][t]
            c3 = params['c3'][t]

            for i in range(N):
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                v[i] = w * v[i] + c1 * r1 * (Xpbest[i] - x[i]) + c2 * r2 * (Xgbest - x[i]) + c3 * r3 * (Xpbest[i] - x[i])
                x[i] += v[i]
                
                for j in range(dim):
                    if j % 3 < 2:
                        x[i][j] = max(0, min(x[i][j], size - 1))
                    else:
                        x[i][j] = max(100, min(x[i][j], 500))
            
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            
            for h in range(N):
                base_stations = [init_base_station(x[h, j*3+2], (x[h, j*3], x[h, j*3+1])) for j in range(num_base_stations)]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[h] = objective_function(size, grid, coverage_grid, special_area)
                
                if fitness[h] < fpbest[h]:
                    fpbest[h] = fitness[h]
                    Xpbest[h] = x[h].copy()
            
            if fpbest.min() < fgbest:
                fgbest = fpbest.min()
                Xgbest = Xpbest[fpbest.argmin()].copy()
            
            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: {fgbest}")
            
            fitness_history.append(fgbest)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        grid = np.zeros((size, size), dtype=float)
        coverage_grid = np.zeros((size, size), dtype=int)
        base_stations = [init_base_station(Xgbest[j*3+2], (Xgbest[j*3], Xgbest[j*3+1])) for j in range(num_base_stations)]
        update_coverage(grid, coverage_grid, base_stations, size)
        total_coverage, special_coverage, general_coverage, std_special, std_general = calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area)
        
        result = {
            'Iteration': tim + 1,
            'Fitness': fgbest,
            'Total Coverage': total_coverage,
            'Special Area Coverage': special_coverage,
            'General Area Coverage': general_coverage,
            'Special Area Signal Uniformity': std_special,
            'General Area Signal Uniformity': std_general,
            'Time': total_time,
            'Base Station Positions': ";".join([f"({Xgbest[j]:.2f}, {Xgbest[j+1]:.2f}, {Xgbest[j+2]:.2f})" for j in range(0, len(Xgbest), 3)])
        }
        
        # 添加空列分隔符
        result[''] = None
        # 添加迭代目标函数值
        for t, value in enumerate(fitness_history):
            result[f"time {t + 1}"] = value
        
        all_results.append(result)

    return all_results
    
def read_parameters(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(':')
            if len(parts) == 2:
                param_name = parts[0].strip()
                values = list(map(float, parts[1].strip().split(',')))
                params[param_name] = values
    return params

file_path = r"D:\桌面\M_PSO\iterations.txt"
params = read_parameters(file_path)

if __name__ == "__main__":
    p = canshu()
    results = run_ahdpso_multiple_times(p, times=5)

    df = pd.DataFrame(results)
    df.to_excel("PSO-4-35.xlsx", index=False)
