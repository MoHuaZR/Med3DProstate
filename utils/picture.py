import matplotlib.pyplot as plt

class DynamicPlot:
    def __init__(self):
        self.loss_values = []
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_title('Loss Over Time')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

    def update_loss(self, loss_value):
        self.loss_values.append(loss_value)

    def plot_loss(self):
        epochs = range(1, len(self.loss_values) + 1)
        self.ax.plot(epochs, self.loss_values, 'b-')
        plt.pause(0.1)
        plt.show()

# 使用示例
dynamic_plotter = DynamicPlot()
current_loss = 0.5  # 替换为实际的损失函数值

# 模拟多次迭代，每次更新一次损失值
for _ in range(5):
    dynamic_plotter.update_loss(current_loss / (_ + 1))

# 调用函数展示动态更新的图表
dynamic_plotter.plot_loss()