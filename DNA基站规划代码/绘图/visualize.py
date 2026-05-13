import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle, Rectangle
import pandas as pd

def init_base_station(height, position, angle_of_view=75):
    radius = (height / 100) * np.tan(np.radians(angle_of_view))
    return {'height': height, 'position': list(position), 'radius': radius}

def get_signal_strength(bs, x, y):
    distance_proj = np.sqrt((x - bs['position'][0]) ** 2 + (y - bs['position'][1]) ** 2)  # 水平距离
    if distance_proj > bs['radius']:  # 未被覆盖
        return 0
    distance = np.sqrt(distance_proj ** 2 + (bs['height'] / 100) ** 2)  # 距离
    loss = 20 * np.log10(distance * 100) + 20 * np.log10(2.4) + 32.45  # 信号衰减dBm （7.604）
    signal_strength_dBm = 20 - loss
    if signal_strength_dBm < -80:
        return 0
    else:
        signal_strength_mw = 10 ** ((signal_strength_dBm - 30) / 10)  # 信号衰减毫瓦mw
        return signal_strength_mw

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

# 可视化
def visualize(base_stations, special_area, size, algorithm_name, block=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制特殊区域
    for i, j in special_area:
        rect = Rectangle((i, j), 1, 1, linewidth=1, edgecolor='blue', facecolor='blue', alpha=1)
        ax.add_patch(rect)

    # 加载并设置自定义图标
    icon_path = r"E:\ZJNU\个人\M_PSO\uav.png"
    if os.path.exists(icon_path):
        icon = plt.imread(icon_path)
    else:
        icon = None

    for bs in base_stations:
        if icon is not None:
            im = OffsetImage(icon, zoom=0.2)  # 调整zoom参数改变图标的大小
            ab = AnnotationBbox(im, bs['position'], xycoords='data', frameon=False)
            ax.add_artist(ab)
        circle = Circle(bs['position'], bs['radius'], color='black', fill=False, linestyle='--', linewidth=1)  # 使用虚线表示覆盖半径
        ax.add_patch(circle)
        ax.plot(bs['position'][0], bs['position'][1], 'ro')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0, size+1, 5))
    ax.set_yticks(np.arange(0, size+1, 5))
    ax.set_title(f'{algorithm_name}')
    
    # 获取桌面路径并保存图像
    desktop = os.path.join(os.path.expanduser("~"), 'Desktop')
    save_path = os.path.join(desktop, f'{algorithm_name}.png')
    plt.savefig(save_path)  # 保存图像到桌面
    
    if block:
        plt.show()  # 阻塞模式，等待用户关闭窗口
    else:
        plt.show(block=False)  # 非阻塞模式
        plt.pause(5)
        plt.close(fig)

def load_special_area():
    filename = r"E:\M_PSO\special_area.txt"
    with open(filename, 'r', encoding='utf-8') as file:
        special_area = json.load(file)
        print(f"Loaded special area from {filename}")
        return special_area

# 读取基站数据
def read_best_base_stations(file_path):
    df = pd.read_excel(file_path)
    min_fitness_row = df.loc[df['Fitness'].idxmin()]
    
    base_station_str = min_fitness_row['Base Station Positions']
    base_stations = []
    for bs_str in base_station_str.split(';'):
        bs_str = bs_str.strip()  # 去除前后空格
        if not bs_str:
            continue
        try:
            # 再次去除括号并分割
            parts = bs_str.strip('()').replace(' ', '').split(',')  # 去掉所有空格再分割
            if len(parts) != 3:
                raise ValueError(f"Invalid format for base station data: {bs_str}")
            x, y, z = map(float, parts)
            bs = init_base_station(height=z, position=(x, y), angle_of_view=75)
            base_stations.append(bs)
        except Exception as e:
            print(f"Error parsing base station '{bs_str}': {e}")
    return base_stations

if __name__ == "__main__":
    special_area = load_special_area()
    excel_file = input("Excel表格地址:")
    base_stations = read_best_base_stations(excel_file)
    algorithm_name = os.path.splitext(os.path.basename(excel_file))[0]

    size = 50
    visualize(base_stations, special_area, size, algorithm_name, block=True)
