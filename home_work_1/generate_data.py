"""
此文件为数据生成函数文件
"""
import math

import numpy as np


# 根据概率生成
def generate_by_p(m_all, p_all, cov, data_size):
    data_list = []
    target_list = []
    print(cov)
    for i in range(len(m_all)):
        size = math.floor(data_size * p_all[i])
        data_list.append(np.random.multivariate_normal(m_all[i], cov, data_size))
        target_list.append([i + 1] * size)
    return data_list, target_list


class GenerateData:
    def __init__(self, default_mean, default_cov, default_size, probabilities_array: list):
        self.default_mean = default_mean
        self.default_cov = default_cov
        self.default_size = default_size
        self.probabilities_array = probabilities_array
        self.plot_data_array = []
        self.real_data_array = []
        self.target_value_array = []
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
