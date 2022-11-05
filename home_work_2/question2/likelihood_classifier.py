"""
用于提供各种计算的类
"""
import numpy as np

from image_plot import ImagePlot


class LikelihoodClassifier(ImagePlot):
    def __init__(self, train_data, train_label, mu, cov):
        self.mu = mu
        self.cov = cov
        self.X_train = train_data
        self.y_train = train_label
        self.p = 1 / len(mu)

    def calculate_g_k(self, x, m, cov) -> float:
        """
        计算第i个数据分布Xi下，第j个数据的gk(X)
        :param x: 当前分类点
        :param m: 均值向量
        :param cov: 协方差矩阵
        :return: g_j(x)
        """
        cov_inv = np.linalg.inv(cov)
        x_mean = x - m
        temp = np.matmul(x_mean, cov_inv)
        temp = np.matmul(temp, x_mean.T)
        # 高斯分布的公式为：1/((2*pi)^(d/2)*|cov|^(1/2))*exp(-1/2*(x-mu)^T*cov^(-1)*(x-mu))
        # 这里因为1/(2*pi)^(d/2) 对于所有的分类都是一样的常数，所以可以省略
        # 但此处不可省略1 / |cov|^(1/2) 因为cov不同，所以需要分别计算
        return np.exp(-0.5 * temp) / np.sqrt(np.linalg.det(cov))

    def calculate_predict(self, x):
        """
        计算x的预测分类
        :return: 预测分类
        """
        g_k = np.zeros(len(self.mu))
        for i in range(len(self.mu)):
            g_k[i] = self.calculate_g_k(x, self.mu[i], self.cov[i])
        return np.argmax(g_k)

    def predict(self, X_test):
        """
        对数据集进行预测
        """
        return np.apply_along_axis(self.calculate_predict, 1, X_test)
