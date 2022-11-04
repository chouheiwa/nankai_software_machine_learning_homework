import os

import numpy as np
from gmm_em import Record


class GenerateData:
    m1 = np.array([0.1, 0.1])
    m2 = np.array([2.1, 1.9])
    m3 = np.array([-1.5, 2.0])
    COV = np.array([np.matrix([[1.2, 0.4],
                               [0.4, 1.8]])] * 3)

    all_mean = np.array([m1, m2, m3])
    P = None
    result = []
    result_target = []

    def __init__(self, load_data=False):
        """
        初始化函数
        :param load_data: 是否加载本地数据(加载数据主要是方便后续的算法分析)
        """
        if load_data and os.path.exists(os.path.join('data', 'data.npy')) and os.path.exists(
                os.path.join('data', 'target.npy')):
            self.result = np.load(os.path.join('data', 'data.npy'))
            self.result_target = np.load(os.path.join('data', 'target.npy'))
        else:
            self.generate_data_method2()
            if not os.path.exists('data'):
                os.makedirs('data')
            np.save(os.path.join('data', 'data.npy'), self.result)
            np.save(os.path.join('data', 'target.npy'), self.result_target)

        self.P = np.zeros(len(self.all_mean))

        for item in self.result_target:
            self.P[item] += 1

        self.P /= len(self.result_target)

    # 根据均值矢量以及协方差使用正态分布生成数据集函数
    def generate_value(self, means, cov, count=1):
        return np.random.multivariate_normal(means, cov, count)

    def generate_data(self):
        """
        第一种生成方法，其核心思想是按照生成规则，顺序生成
        :rtype: 样本
        """
        result = []
        result_target = []
        for i in range(500):
            current_type = i % 4  # 生成规则为4个一组

            if current_type < 2:  # 前2个
                result_target.append(1)  # 本次代码中会将题目中的数字角标减1，使其从由 [1, 2, 3] -> [0, 1, 2]
            elif current_type == 2:  # 第三个
                result_target.append(0)
            else:  # 最后一个
                result_target.append(2)
            result.append(self.generate_value(self.all_mean[result_target[i]], self.COV[result_target[i]]))

        self.result = np.concatenate(result, axis=0)
        self.result_target = np.array(result_target)

    def generate_data_method2(self):
        """
        第二种生成方式，其为总体生成，即样本生成后其总体数值进行直接生成。
        """
        result = []
        result_target = []
        counts = [125, 250, 125]
        for i in range(3):
            result.append(self.generate_value(self.all_mean[i], self.COV[i], counts[i]))
            result_target.extend([i] * counts[i])

        self.result = np.concatenate(result, axis=0)
        self.result_target = np.array(result_target)

    def get_target_record(self):
        return Record(mu=self.all_mean, cov=self.COV, pi=self.P)
