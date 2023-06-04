import matplotlib.pyplot as plt


def draw(losses):
    # 创建横坐标（迭代次数）和纵坐标（损失值）的数据
    x = range(1, len(losses) + 1)
    y = losses

    # 绘制损失函数曲线
    plt.plot(x, y, 'b-')

    # 设置图表标题和轴标签
    plt.title('Loss Function Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # 显示网格线
    plt.grid(True)

    # 显示图表
    plt.show()
