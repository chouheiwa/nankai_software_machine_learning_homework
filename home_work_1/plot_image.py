import matplotlib.pyplot as plt
import numpy as np

colors = ['blue', 'orange', 'green', 'red']


# 添加绘图功能
def plot_data(result, result_error, name, file_name=None):
    """
    将给定的数据集绘制出散点图
    :param result: 绘制分类数据集
    :param result_error: 实验出错的点
    :param name: 数据图像标题名称
    :param file_name: 保存图片文件
    """
    fig = plt.figure(figsize=(12, 12))
    plt.axis('equal')
    for i in range(len(result)):
        if len(result[i]) == 0:
            continue
        plt.plot(result[i][:, 0], result[i][:, 1], 'o', markersize=2, mec=colors[i])
    if result_error is not None and len(result_error) != 0:
        plt.plot(result_error[:, 0], result_error[:, 1], 'o', markersize=2, mec=colors[-1])
    plt.title(f"{name} data")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    if file_name is not None:
        plt.show()
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
    x = np.arange(-4, 12, 0.05)
    y = np.arange(-6, 10, 0.05)
    x, y = np.meshgrid(x, y)
    for i in range(x.shape[0]):
        result.append([])
        for j in range(x.shape[1]):
            temp_1 = calculate_function([x[i][j], y[i][j]], 0)
            temp_2 = calculate_function([x[i][j], y[i][j]], 1)
            temp_3 = calculate_function([x[i][j], y[i][j]], 2)
            result[i].append([temp_1, temp_2, temp_3])

    for item in index_array:
        z = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i][j] = result[i][j][item[0]] - result[i][j][item[1]] + (
                    1000 if result[i][j][item[2]] >= result[i][j][item[0]] and result[i][j][
                        item[2]] >= result[i][j][item[1]] else 0)
        plt.contour(x, y, z, 0)
    plt.show()
    plt.close()
    fig.savefig(file_name)
