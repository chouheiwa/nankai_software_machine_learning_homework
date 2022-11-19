import os

import numpy as np
import plot_image as pi


class GenerateData:
    size = 10
    n_mean = [0]
    n_cov = [0.04]

    noises = None
    target_x = None
    target_y = None

    def __init__(self, load_data=False):
        """
        初始化函数
        :param load_data: 是否加载本地数据(加载数据主要是方便后续的算法分析)
        """
        dir_path = pi.get_path('data')
        if load_data and os.path.exists(os.path.join(dir_path, 'noise.npy')):
            self.noises = np.load(os.path.join(dir_path, 'noise.npy'))
        else:
            self.noises = self.generate_value(self.n_mean, self.n_cov, self.size)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            np.save(os.path.join(dir_path, 'noise.npy'), self.noises)

        self.target_x = np.arange(-5, 5)
        self.target_y = self.function_x(self.target_x) + self.noises

    # 根据均值矢量以及协方差使用正态分布生成数据集函数
    def generate_value(self, means, cov, count=1):
        return np.random.normal(means, cov, count)

    def function_x(self, target_x):
        return -np.sin(target_x / 5) + np.cos(target_x)
