"""
此文件为数据生成函数文件
"""
import math

import numpy as np


# 根据均值和协方差的生成正态分布的数据
def generate_by_p(m_all, p_all, cov, data_size):
    data_list = []
    target_list = []
    for i in range(len(m_all)):
        size = math.floor(data_size * p_all[i])
        data_list.append(np.random.multivariate_normal(m_all[i], cov, size))
        target_list.append([i + 1] * size)
    return data_list, target_list


class GenerateData:
    def __init__(self, default_mean, default_cov, default_size, probabilities_array: list):
        self.default_mean = default_mean  # 默认均值数组
        self.default_cov = default_cov  # 默认协方差
        self.default_size = default_size  # 数据总体大小
        self.probabilities_array = probabilities_array  # 数据的先验概率
        self.plot_data_array = []  # 用于绘制散点图的数据
        self.real_data_array = []  # 将散点图数组展开后的数据
        self.target_value_array = []  # 将散点图数组展开后的数据的标签
        self.generate()

    def generate(self):
        self.plot_data_array = []
        self.real_data_array = []
        self.target_value_array = []
        for index, probability in enumerate(self.probabilities_array):
            temp_X, temp_y = generate_by_p(self.default_mean, probability, self.default_cov, self.default_size)
            self.plot_data_array.append(temp_X)
            self.real_data_array.append(np.concatenate(temp_X, axis=0))
            self.target_value_array.append([i for j in temp_y for i in j])
