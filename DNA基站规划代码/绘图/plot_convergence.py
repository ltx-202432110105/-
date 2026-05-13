import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_fitness_data(file_path, target_iteration):
    df = pd.read_excel(file_path)
    target_row = df[df['Iteration'] == target_iteration]
    if target_row.empty:
        raise ValueError(f"No data found for Iteration {target_iteration} in {file_path}")
    
    num_groups = 10
    group_size = 50
    fitness_values = target_row.iloc[0, 10:510].values
    averaged_fitness = [fitness_values[0]]
    for i in range(num_groups):
        averaged_fitness.append(np.mean(fitness_values[i * group_size:(i + 1) * group_size]))

    return averaged_fitness

# 计算横坐标
def calculate_x_coordinates():
    return [float(i / 10 * 1.5) for i in range(11)]

# 绘制 Fitness 收敛曲线
def plot_convergence(averaged_fitness, x_coordinates, label, marker):
    plt.plot(x_coordinates[:11], averaged_fitness, marker=marker, label=label)


file_paths = [r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\AHDPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\AFMPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\AMSEPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\FHPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\SAOMPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\SDPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\SPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\MPSO1.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\MPSO.xlsx",
                r"E:\ZJNU\个人\M_PSO\相同评估次数\不固定\PSO.xlsx"
]

algorithm_names = ['AHDPSO', 'AFMPSO', 'AMSEPSO','FHPSO','SAOMPSO','SDPSO','SPSO','MPSO','MBPSO','PSO']
# ['s', 'o', '^', 'v', '<', '>', 'p', '*', 'h', '#', '+', 'x', 'D', 'd', '|', '_']
markers = ['s','o','^','v','<','>','p','*','h','D']
target_iteration = 21
    
x_coordinates = calculate_x_coordinates()
    
for file_path, algorithm_name, marker in zip(file_paths, algorithm_names, markers):
    try:
        averaged_fitness = extract_fitness_data(file_path, target_iteration)
        plot_convergence(averaged_fitness, x_coordinates, algorithm_name, marker)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    
plt.title("Fitness Convergence")
plt.xlabel("Evaluation Count")
plt.ylabel("Average Fitness Value")
plt.xlim(-0.03, max(x_coordinates)+0.03)
plt.xticks(x_coordinates)
plt.ylim(0,0.22)
plt.yticks(np.arange(0, 0.22, 0.05))
plt.legend()
plt.grid(True)
plt.text(max(x_coordinates)+0.15, min(plt.ylim()), r'$\times 10^4$',  ha='right', va='top')
plt.show()
