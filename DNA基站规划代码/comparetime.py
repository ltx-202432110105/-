import matplotlib.pyplot as plt

algorithms = ['AHDPSO', 'AFMPSO', 'AMSEPSO','FHPSO','SAOMPSO','SDPSO','SPSO','MPSO1','MPSO','PSO']  # 算法名称
fixed_times = [2065.48,2279.76,4297.75,6135.33,2463.85,6339.23,2337.49,6344.98,1782.59,1829.40]
random_times = [2512.42, 2364.05, 4455.87,6312.45,2374.26,6500.36,2457.16,6362.24,2416.44,2440.56]

# 设置柱状图的位置和宽度
bar_width = 0.35
index = range(len(algorithms))

# 创建柱状图，并设置图形大小
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar([i - bar_width/2 for i in index], fixed_times, bar_width, label='Fixed')
rects2 = ax.bar([i + bar_width/2 for i in index], random_times, bar_width, label='Random')

# 设置图表标题和坐标轴标签
ax.set_xlabel('Algorithms')
ax.set_ylabel('Time/s')
ax.set_title('Run time Comparison of Different Algorithms')
ax.set_xticks(index)
ax.set_xticklabels(algorithms)
ax.legend()

# 显示图表
plt.tight_layout()
plt.show()