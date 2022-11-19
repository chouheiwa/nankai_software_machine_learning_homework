import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

colors = ['blue', 'orange', 'green', 'red']
font = FontProperties(fname=r"SimHei.ttf", size=14)


def get_path(file):
    """
    获取文件路径 用于兼容不同目录下执行此python文件可能导致路径读取错误的问题
    :param file:
    :return:
    """
    return path.join(path.dirname(path.abspath(__file__)), file)


# 添加绘图功能
def plot_data(x, y, function_x, plot_image=True):
    """
    将给定的数据集绘制出散点图
    :param result: 绘制分类数据集
    :param result_class: 绘制分类数据集的类别(从0开始)
    """
    fig = plt.figure()
    # plt.axis('equal')
    plot_item, = plt.plot(x, y, 'o', markersize=4, mec=colors[0])

    plt.title(f"数据散点及函数分布图", fontproperties=font)
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    dir_path = get_path('images')
    if not path.exists(dir_path):
        os.makedirs(dir_path)

    default_x = np.arange(-5, 5, 0.01)
    default_y = function_x(default_x)
    plt.plot(default_x, default_y)

    plt.grid()
    if plot_image:
        fig.savefig(path.join(dir_path, 'data.png'))
        plt.close()

    return fig


def plot_gradient_history(history):
    """
    绘制梯度下降的历史记录
    :param history:
    :return:
    """
    fig = plt.figure()
    plt.title(f"梯度下降历史记录", fontproperties=font)
    plt.xlabel("迭代次数", fontproperties=font)
    plt.ylabel("梯度值", fontproperties=font)

    dir_path = get_path('images')
    if not path.exists(dir_path):
        os.makedirs(dir_path)

    plt.plot(np.arange(len(history)), history)

    plt.grid()
    fig.savefig(path.join(dir_path, 'gd_history.png'))
    plt.close()


def plot_predict_line(W, x, y, function_x):
    """
    绘制预测直线
    :param W:
    :return:
    """
    fig = plot_data(x, y, function_x, plot_image=False)
    plt.title(f"预测结果", fontproperties=font)

    dir_path = get_path('images')
    if not path.exists(dir_path):
        os.makedirs(dir_path)

    param_length = W.shape[0] - 1

    default_x = np.arange(-5, 5, 0.01)

    data_array = [np.ones(default_x.shape[0])]

    for i in range(param_length):
        data_array.append(default_x ** (i + 1))

    default_X = np.array(data_array).T

    predict_y = np.matmul(default_X, W)

    plt.plot(default_x, predict_y)

    # plt.grid()
    fig.savefig(path.join(dir_path, f'predict_line_poly_{param_length}.png'))
    plt.close()
