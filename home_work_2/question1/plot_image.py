import os
from os import path
from gmm_em import Record, AllRecord

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
def plot_data(result, result_class):
    """
    将给定的数据集绘制出散点图
    :param result: 绘制分类数据集
    :param result_class: 绘制分类数据集的类别(从0开始)
    """
    fig = plt.figure(figsize=(12, 12))
    plt.axis('equal')
    plot_array = []
    plot_title = []

    group = []
    for i in range(result.shape[0]):
        count = result_class[i] + 1 - len(group)
        while count > 0:
            group.append(np.ndarray(shape=(0, 2)))
            count -= 1
        group[result_class[i]] = np.concatenate((group[result_class[i]], np.array([result[i]])), axis=0)
        # group[result_class[i]].append([result[i][0], result[i][1]])

    for i in range(len(group)):
        if len(group[i]) == 0:
            continue
        plot_item, = plt.plot(group[i][:, 0], group[i][:, 1], 'o', markersize=4, mec=colors[i])
        plot_array.append(plot_item)
        plot_title.append(f'$m_{i + 1}$: {len(group[i])}')
    plt.legend(plot_array, plot_title)
    plt.title(f"数据散点图", fontproperties=font)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    dir_path = get_path('images')
    if not path.exists(dir_path):
        os.makedirs(dir_path)
    # plt.show()
    fig.savefig(path.join(dir_path, 'data.png'))
    plt.close()

    return fig


def plot_record(record: AllRecord, real: Record, order):
    """
    绘制记录
    :param record: 记录
    :param real: 相关生成条件
    :param order:
    """
    fig = plt.figure(figsize=(12, 12))

    x = np.arange(len(record.record_list))
    mu_indicator = record.get_mu_indicator()[:, order.tolist()]
    for i in range(len(real.pi)):
        for j in range(2):
            ax = plt.subplot(len(real.pi), 2, i * 2 + j + 1, label=f'Record {i + 1}')
            ax.plot(x, mu_indicator[:, i, j])
            ax.set_title(f'第{i + 1}个高斯分布期望的第{j + 1}个参数随迭代次数的变化', fontproperties=font)

    dir_path = get_path('images')
    if not path.exists(dir_path):
        os.makedirs(dir_path)
    # plt.show()
    fig.savefig(path.join(dir_path, 'mu_iteration.png'))
    plt.close()

    return fig
