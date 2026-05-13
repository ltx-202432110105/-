import time
import numpy as np
import pandas as pd
from tetromino_generator import generate_special_area
from model import canshu, init_base_station, update_coverage, objective_function, calculate_coverage_and_uniformity, load_special_area

def run_saompso_multiple_times(p, times):
    all_results = []
    for tim in range(times):
        print(f"Running iteration {tim + 1}/{times}")
        
        size, num_base_stations, N, iterators = p['size'], p['num_base_stations'], p['num_particles'], p['max_iterations']

        special_area = generate_special_area(size=p['size'])

        # 初始化种群个体、移动速度以及适应度值
        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)] for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N
        
        Xgbest = x[fitness.index(min(fitness))].copy()
        fgbest = min(fitness)
        Xpbest = [xi.copy() for xi in x]
        fpbest = fitness.copy()
        
        start_time = time.time()
        fitness_history = []

        for t in range(iterators):
            w = 0.3 * np.sin((2 * np.pi * t) / iterators) + 0.6
            c1 = np.sin((3 * np.pi * t) / iterators) + 1.5
            c2 = np.cos((3 * np.pi * t) / iterators) + 1.5
            
            for i in range(N):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                T = np.exp(-t / iterators)
                k = 1
                DDF = 0.35 * (1 + (5 / 7) * ((np.exp(t / iterators) - 1) ** k) / ((np.exp(1) - 1) ** k))
                M = DDF * T

                for j in range(num_base_stations):
                    for k_dim in range(3):
                        v[i][j][k_dim] = (
                            w * v[i][j][k_dim] +
                            c1 * r1 * (Xpbest[i][j][k_dim] - x[i][j][k_dim]) +
                            c2 * r2 * (Xgbest[j][k_dim] - x[i][j][k_dim]) +
                            M * np.random.randn()
                        )
                    
                    Vmax = 6 * (1 - np.log10(1 + 9 * t / 1000))
                    v[i][j] = np.clip(v[i][j], -Vmax, Vmax)
                    
                    x[i][j] = [x[i][j][k] + v[i][j][k] for k in range(3)]

                for j in range(num_base_stations):
                    x[i][j][0] = max(0, min(x[i][j][0], size - 1))
                    x[i][j][1] = max(0, min(x[i][j][1], size - 1))
                    x[i][j][2] = max(100, min(x[i][j][2], 500))
            
            grid = np.zeros((size, size), dtype=float)
            coverage_grid = np.zeros((size, size), dtype=int)
            
            for i in range(N):
                base_stations = [init_base_station(bs[2], (bs[0], bs[1])) for bs in x[i]]
                update_coverage(grid, coverage_grid, base_stations, size)
                fitness[i] = objective_function(size, grid, coverage_grid, special_area)
                
                if fitness[i] < fpbest[i]:
                    fpbest[i] = fitness[i]
                    Xpbest[i] = [bs.copy() for bs in x[i]]
            
            if min(fpbest) < fgbest:
                fgbest = min(fpbest)
                Xgbest = Xpbest[fpbest.index(fgbest)].copy()

            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: {fgbest}")

            fitness_history.append(fgbest)
        
        end_time = time.time()
        total_time = end_time - start_time
        
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
        
        result[''] = None
        for t, value in enumerate(fitness_history):
            result[f"time {t + 1}"] = value
        
        all_results.append(result)
    
    return all_results

if __name__ == "__main__":
    p = canshu()
    results = run_saompso_multiple_times(p, times=20)

    df = pd.DataFrame(results)
    df.to_excel("SAOMPSO_results1.xlsx", index=False)
