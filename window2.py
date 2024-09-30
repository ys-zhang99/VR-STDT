import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
#设置中文字体
plt.rcParams['axes.unicode_minus'] = False
#设置负号显示

# 生成示例时序数据
np.random.seed(0)
x = np.linspace(0, 10 * np.pi, 500)
data = (np.sin(x) + 0.5 * np.sin(2 * x + np.pi/4) + 0.3 * np.sin(3 * x - np.pi/3)) * 20 + 100 + np.random.normal(0, 5, 500)

# 设置异常波动区域的起始和结束位置
anomaly_start = 400
anomaly_end = 500

# 设置时间窗口的位置
window_start = 430
window_end = 480

fig, ax = plt.subplots(figsize=(12, 6))

# 绘制完整的时序数据
ax.plot(data, color='blue', label='正常数据')

# 绘制异常波动区域
ax.plot(range(anomaly_start, anomaly_end), data[anomaly_start:anomaly_end], color='red', label='异常波动')

# 绘制无法覆盖全部异常波动数据的时间窗口
window_rect = plt.Rectangle((window_start, min(data[anomaly_start:anomaly_end])), 
                            window_end - window_start, 
                            max(data[anomaly_start:anomaly_end]) - min(data[anomaly_start:anomaly_end]), 
                            edgecolor='green', facecolor='none', linestyle='--', linewidth=2)
ax.add_patch(window_rect)

# 添加文本说明
ax.text(anomaly_start + 10, 90, '异常波动', color='red', fontsize=24)
ax.text(440, 70, '时间窗口', color='green', fontsize=24)

# 设置坐标范围和标签
ax.set_xlim([350, 510])
ax.set_ylim([50, 150])
#ax.set_title('时间窗口无法涵盖全部异常波动数据示意图', fontsize=16)
ax.set_xlabel('时间/s', fontsize=18)
ax.set_ylabel('传感数值', fontsize=18)
ax.legend(fontsize=18)

# 放大横纵坐标的刻度文本
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()