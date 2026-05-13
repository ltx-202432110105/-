import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

excel_files = [r"D:\桌面\M_PSO\相同评估次数\固定\AHDPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\AFMPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\AMSEPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\FHPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\SAOMPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\SDPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\SPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\MPSO1.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\MPSO.xlsx",
                r"D:\桌面\M_PSO\相同评估次数\固定\PSO.xlsx"
]

# 读取数据并处理
data = []
for file in excel_files:
    df = pd.read_excel(file)
    data.append(np.abs(np.array(df.iloc[:, 1].values)))

fig, ax = plt.subplots(figsize=(9, 6))

boxprops = dict(linestyle='-', linewidth=1, color='black')
whiskerprops = dict(linestyle='-', linewidth=1, color='black')
capprops = dict(linestyle='-', linewidth=1, color='black')
medianprops = dict(linestyle='-', linewidth=1, color='red')
flierprops = dict(marker='o', markerfacecolor='green', markersize=5, linestyle='none', markeredgecolor='black')
meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick', markersize=5)

bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanprops=meanprops, flierprops=flierprops,
                 boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)

ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")
ax.set_ylabel('Objective Function(%)', rotation=90, labelpad=5)
ax.tick_params(axis='both', labelsize=9)

ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

# 设置x轴标签
ax.set_xticks(range(1, len(excel_files) + 1))
ax.set_xticklabels(['AHDPSO', 'AFMPSO', 'AMSEPSO', 'FHPSO', 'SAOMPSO', 'SDPSO', 'SPSO', 'MPSO1', 'MPSO', 'PSO'])

for patch, color in zip(bp['boxes'], plt.cm.Pastel1.colors):
    new_color = list(color[:6]) + [0.7]
    patch.set_facecolor(new_color)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.1)
ax.spines['left'].set_linewidth(1.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_tick_params(width=1)
ax.yaxis.set_tick_params(width=1)

ax.legend([bp["means"][0], bp["fliers"][0], bp["medians"][0]], ['Mean Value', 'Outliers', 'Median Line'], frameon=False)

plt.show()
