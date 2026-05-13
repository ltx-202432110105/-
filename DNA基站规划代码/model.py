import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle, Rectangle

def canshu():
    return {
        'size': 50,
        'num_base_stations': 30,
        'num_particles': 30,
        'max_iterations': 500
    }

# 初始化基站
def init_base_station(height, position, angle_of_view=75):
    radius = (height / 100) * np.tan(np.radians(angle_of_view))
    return {'height': height, 'position': list(position), 'radius': radius}

# 信号强度
def get_signal_strength(bs, x, y):
    distance_proj = np.sqrt((x - bs['position'][0]) ** 2 + (y - bs['position'][1]) ** 2)  # 水平距离
    if distance_proj > bs['radius']:  # 未被覆盖
        return 0
    distance = np.sqrt(distance_proj ** 2 + (bs['height'] / 100) ** 2)
    loss = 20 * np.log10(distance * 100) + 20 * np.log10(2.4) + 32.45  # 信号衰减dBm
    signal_strength_dBm = 20 - loss
    if signal_strength_dBm < -80:
        return 0
    else:
        signal_strength_mw = 10 ** ((signal_strength_dBm - 30) / 10)  # 信号衰减毫瓦mw
        return signal_strength_mw

# 覆盖区域
def update_coverage(grid, coverage_grid, base_stations, size):
    grid.fill(1e-10)  # 区域信号初始为1e-10
    coverage_grid.fill(0)  # 覆盖区域初始为0
    for i in range(size):
        for j in range(size):
            x_center, y_center = i + 0.5, j + 0.5  # 小方格中心点
            total_power = 1e-10
            for bs in base_stations:
                distance_proj = np.sqrt((x_center - bs['position'][0]) ** 2 + (y_center - bs['position'][1]) ** 2)
                if distance_proj <= bs['radius']:
                    signal_strength = get_signal_strength(bs, x_center, y_center)
                    total_power += signal_strength
            grid[i][j] = total_power
            if total_power > 1e-10:
                coverage_grid[i][j] = 1

# 信号均匀性
def calculate_uniformity(size, grid, special_area):
    def mw_to_dBm(power_mw): # 信号强度毫瓦mw转化为dBm
        return 10 * np.log10(power_mw)
    
    std_special = np.std([mw_to_dBm(grid[i, j]) for i, j in special_area])
    std_general = np.std([mw_to_dBm(grid[i, j]) for i in range(size) for j in range(size) if (i, j) not in special_area])
    return std_special, std_general

# 目标函数值
def objective_function(size, grid, coverage_grid, special_area):
    std_special, std_general = calculate_uniformity(size, grid, special_area)
    f1 = np.sum(coverage_grid == 0) / (size ** 2)
    f2 = np.log(std_special + 1) / np.log(100)
    f3 = np.log(std_general + 1) / np.log(1000)
    F = 0.5 * f1 + 0.3 * f2 + 0.2 * f3
    return F

def calculate_coverage_and_uniformity(size, grid, coverage_grid, special_area):
    total_coverage = np.mean(coverage_grid)
    special_coverage = np.mean(coverage_grid[special_area]) if special_area else 0
    general_coverage = np.mean(coverage_grid[~np.isin(np.arange(size**2).reshape(size, size), special_area)])
    std_special, std_general = calculate_uniformity(size, grid, special_area)
    print(f"Total Coverage: {total_coverage:.2%}")
    print(f"Special Area Coverage: {special_coverage:.2%}")
    print(f"General Area Coverage: {general_coverage:.2%}")
    print(f"Special Area Signal Uniformity: {std_special:.2f} dB")
    print(f"General Area Signal Uniformity: {std_general:.2f} dB\n")

    return total_coverage, special_coverage, general_coverage, std_special, std_general

# 可视化
def visualize(iteration, base_stations, special_area, size, block):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制特殊区域
    for i, j in special_area:
        rect = Rectangle((i, j), 1, 1, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
        ax.add_patch(rect)

    # 加载并设置自定义图标
    icon_path = r"D:\桌面\M_PSO\uav.png"
    icon = plt.imread(icon_path)

    for bs in base_stations:
        im = OffsetImage(icon, zoom=0.2)  # 调整zoom参数改变图标的大小
        ab = AnnotationBbox(im, bs['position'], xycoords='data', frameon=False)
        ax.add_artist(ab)
        circle = Circle(bs['position'], bs['radius'], color='yellow', fill=False, linestyle='-', linewidth=1)
        ax.add_patch(circle)
        ax.plot(bs['position'][0], bs['position'][1], 'ro')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0, size+1, 5))
    ax.set_yticks(np.arange(0, size+1, 5))
    ax.set_title(f'Iteration {iteration + 1}')
    
    # 获取桌面路径并保存图像
    desktop = os.path.join(os.path.expanduser("~"), 'Desktop')
    save_path = os.path.join(desktop, f'iteration_{iteration + 1}.png')
    plt.savefig(save_path)  # 保存图像到桌面
    
    if block:
        plt.show()  # 阻塞模式，等待用户关闭窗口
    else:
        plt.show(block=False)  # 非阻塞模式
        plt.pause(5)
        plt.close(fig)

def final_pic(max_iterations, fitness_history):
    plt.figure()
    plt.plot(range(1, max_iterations + 1), fitness_history)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Objective Function Value vs Iteration') 
    plt.show()


def load_special_area():
    filename = r"D:\桌面\M_PSO\special_area.txt"
    with open(filename, 'r', encoding='utf-8') as file:
        special_area = json.load(file)
        print(f"Loaded special area from {filename}")
        return special_area
