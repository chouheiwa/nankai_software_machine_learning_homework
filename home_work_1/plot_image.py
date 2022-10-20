import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

colors = ['blue', 'orange', 'green', 'red']
font = {"family": "KaiTi", "size": 14}  # 设置字体


# 添加绘图功能
def plot_data(result, result_error, name, file_name=None):
    """
    将给定的数据集绘制出散点图
    :param result: 绘制分类数据集
    :param result_error: 实验出错的点
    :param name: 数据图像标题名称
    :param file_name: 保存图片文件
    """
    fig = plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plot_array = []
    plot_title = []
    for i in range(len(result)):
        if len(result[i]) == 0:
            continue
        plot_item, = plt.plot(result[i][:, 0], result[i][:, 1], 'o', markersize=4, mec=colors[i])
        plot_array.append(plot_item)
        plot_title.append(f'$m_{i + 1}$')
    if result_error is not None and len(result_error) != 0:
        plot_item, = plt.plot(result_error[:, 0], result_error[:, 1], 'o', markersize=4, mec=colors[-1])
        plot_array.append(plot_item)
        plot_title.append('error')
    plt.legend(plot_array, plot_title)
    plt.title(f"{name} 数据图", fontdict=font)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    if file_name is not None:
        # plt.show()
        fig.savefig(file_name)
        plt.close()
    return fig


def plot_data_line(result, result_error, name, file_name, calculate_function):
    """
    将给定的数据集绘制出散点图，并且绘制出分类线
    :param result: 绘制分类数据集
    :param result_error: 实验出错的点
    :param name: 数据图像标题名称
    :param file_name: 保存图片文件
    :param calculate_function: 计算函数，入参为x的向量，与当前均值索引，返回值为函数计算数值
    """
    fig = plot_data(result, result_error, name, None)

    index_array = [[0, 1, 2], [0, 2, 1], [1, 2, 0]]
    result = []
    x = np.arange(-4, 12, 0.1)
    y = np.arange(-6, 10, 0.1)
    x, y = np.meshgrid(x, y)
    for i in range(x.shape[0]):
        result.append([])
        for j in range(x.shape[1]):
            temp_1 = calculate_function(np.array([x[i][j], y[i][j]]), 0)
            temp_2 = calculate_function(np.array([x[i][j], y[i][j]]), 1)
            temp_3 = calculate_function(np.array([x[i][j], y[i][j]]), 2)
            result[i].append([temp_1, temp_2, temp_3])

    for item in index_array:
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                temp_1 = result[i][j][item[0]]
                temp_2 = result[i][j][item[1]]
                temp_3 = result[i][j][item[2]]
                max_data = max(temp_1, temp_2, temp_3)

                z[i][j] = temp_1 - temp_2
                # if max_data == temp_3:
                #     z[i][j] += 100

        plt.contour(x, y, z, 0)
    # plt.show()
    plt.close()
    fig.savefig(file_name)
