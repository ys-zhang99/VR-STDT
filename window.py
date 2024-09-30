import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 生成示例时序数据
np.random.seed(0)
x = np.linspace(0, 10 * np.pi, 500)
data = (np.sin(x) + 0.5 * np.sin(2 * x + np.pi/4) + 0.3 * np.sin(3 * x - np.pi/3)) * 20 + 100 + np.random.normal(0, 5, 500)

# 设置窗口的大小和移动步长
window_size = 50
step_size = 5

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# 绘制完整的时序数据
ax1.plot(data, color='blue')
window_rect = plt.Rectangle((0, 50), window_size, 100, edgecolor='red', facecolor='none')
ax1.add_patch(window_rect)
ax1.set_title('Complete Time Series')
ax1.set_xlim([0, len(data)])
ax1.set_ylim([50, 150])
ax1.set_xlabel('Time/s')
ax1.set_ylabel('Sensor Data')

# 设置下方放大的窗口
line, = ax2.plot(data[:window_size], color='red')
ax2.set_title('Zoomed Window')
ax2.set_xlim([0, window_size])
ax2.set_ylim([50, 150])
ax2.set_xlabel('Time/s')
ax2.set_ylabel('Sensor Data')

# 更新函数
def update(frame):
    start = frame * step_size
    end = start + window_size
    window_rect.set_x(start)
    line.set_ydata(data[start:end])
    return window_rect, line

# 动画设置
ani = animation.FuncAnimation(fig, update, frames=range((len(data) - window_size) // step_size), blit=True, interval=100)

plt.tight_layout()
plt.show()