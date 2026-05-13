import numpy as np
from model import canshu

c1_max = c2_max = 2
c1_min = c2_min = 0
c3_max = 1
c3_min = 0
a = 0.8 # 振荡幅度amplitude
u = 0.6 # 系数变化速度
max_iterations = canshu()['max_iterations']
r = np.zeros(max_iterations)
r[0] = np.random.rand() 
for t in range(1, max_iterations):
    r[t] = 4 * r[t-1] * (1 - r[t-1])

def generate_c1(max_iterations):
    c1 = np.zeros(max_iterations)
    for t in range(max_iterations):
        c1[t] = max - a - u * (max - min) * (t / max_iterations) + (1 - t / max_iterations) * r[t] * a
    return c1

def generate_c2(max_iterations):
    c2 = np.zeros(max_iterations)
    for t in range(max_iterations):
        c2[t] = max - a - u * (max - min) * ((max_iterations-t-1) / max_iterations) + (1 - (max_iterations-t-1) / max_iterations) * r[t] * a
    return c2

def generate_c3(max_iterations):
    c3 = np.zeros(max_iterations)
    for t in range(max_iterations):
        if t < max_iterations/2:
            c3[t] = (c3_max - c3_min) * (t / max_iterations) + (t / max_iterations) * r[t] * a
        else:
            c3[t] = (c3_max - c3_min) * ((max_iterations - t - 1) / max_iterations) + ((max_iterations - t - 1) / max_iterations) * r[t] * a
    return c3

def generate_w(max_iterations):
    w_max = 0.9
    w_min = 0.4
    return [round(w_max - (w_max - w_min) / max_iterations * i, 3) for i in range(max_iterations)]

c1_values = generate_c1(max_iterations)
c2_values = generate_c2(max_iterations)
c3_values = generate_c3(max_iterations)
w_values = generate_w(max_iterations)

# 结果保存到指定文件中，每次写入覆盖之前的内容
file_path = r"D:\桌面\分裂粒子群代码\M_PSO\iterations.txt"
with open(file_path, 'w') as file:
    for name, values in zip(['c1', 'c2', 'c3', 'w'], [c1_values, c2_values, c3_values, w_values]):
        # 写入文件
        file.write(f"{name}: {','.join(map(str, values))}\n")

print(f"Lists have been saved to {file_path}")

# 绘制图像
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(c1_values, label=r'C$_1$', color='blue')
axs[0].plot(c2_values, label=r'C$_2$', color='orange')
axs[0].set_title(r'C$_1$ and C$_2$ Values Over Iterations')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Value')
axs[0].legend()

axs[1].plot(c3_values, label=r'C$_3$', color='red')
axs[1].set_title(r'C$_1$ Values Over Iterations')
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel('Value')
axs[1].legend()

plt.tight_layout()
plt.show()

