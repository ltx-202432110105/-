import time
import numpy as np
import pandas as pd
from tetromino_generator import generate_special_area
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

def run_sdpso_multiple_times(p, times):
    """运行SDPSO算法多次"""
    all_results = []
    
    for tim in range(times):
        print(f"Running SDPSO iteration {tim + 1}/{times}")
        
        size = p['size']
        num_base_stations = p['num_base_stations']
        N = p['num_particles']
        iterators = p['max_iterations']
        special_area = generate_special_area(size=p['size'])
        
        # 初始化参数
        initial_temp = 1000  # 模拟退火初始温度
        m_ratio = 0.3       # DLS阈值比例
        m_threshold = int(m_ratio * iterators)
        
        # 初始化种群
        x = [[list(np.random.uniform(0, size, 2)) + [np.random.uniform(100, 500)] 
              for _ in range(num_base_stations)] for _ in range(N)]
        v = [[[0, 0, 0] for _ in range(num_base_stations)] for _ in range(N)]
        fitness = [float('inf')] * N
        
        # 惯性权重范围
        w_max = 0.9
        w_min = 0.4
        
        # 全局和个体最优初始化
        Xgbest = x[fitness.index(min(fitness))].copy()
        fgbest = min(fitness)
        Xpbest = [xi.copy() for xi in x]
        fpbest = fitness.copy()
        
        start_time = time.time()
        fitness_history = []
        
        for t in range(iterators):
            # 动态参数调整
            w = w_max - (w_max - w_min) * (t / iterators)  # 惯性权重
            temperature = initial_temp * (1 - t/iterators)  # 退火温度
            c1 = 2.0 - (2.0 - 0.4) * (t / iterators)      # 认知因子
            c2 = 0.4 + (2.0 - 0.4) * (t / iterators)      # 社会因子
            
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
                x[i] = dimensional_learning(x[i], Xgbest, m_threshold, t, iterators)
                
                # 边界处理
                for j in range(num_base_stations):
                    x[i][j][0] = np.clip(x[i][j][0], 0, size-1)
                    x[i][j][1] = np.clip(x[i][j][1], 0, size-1)
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
            
            # 记录历史
            fitness_history.append(fgbest)
            
            if (t + 1) % 20 == 0:
                print(f"Iteration {t + 1}: Fitness={fgbest:.4f}, Temp={temperature:.2f}")
        
        # 结果计算
        end_time = time.time()
        total_time = end_time - start_time
        
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
        
        # 记录历史适应度
        for t_val, value in enumerate(fitness_history):
            result[f"time {t_val + 1}"] = value
        
        all_results.append(result)
    
    return all_results

if __name__ == "__main__":
    p = canshu()
    results = run_sdpso_multiple_times(p, times=20)
    
    df = pd.DataFrame(results)
    df.to_excel("SDPSO_result1.xlsx", index=False)
