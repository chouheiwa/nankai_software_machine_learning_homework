# -*- coding:utf-8 -*-
import numpy as np


class Record:
    def __init__(self, mu, cov, pi):
        self.mu = mu
        self.cov = cov
        self.pi = pi


class AllRecord:
    record_list: [Record] = []

    def __init__(self, initial_mu):
        self.initial_mu = initial_mu

    def get_mu(self):
        return np.array([item.mu for item in self.record_list])

    def get_mu_indicator(self):
        """
        这里因为mu为2维，图像渲染时需要将其转换为1维，因此这里简单的使用求和函数将其转为1维，此处仅为了方便图像渲染，用以显示mu是否收敛，无物理方面的意义
        :return:
        """
        return self.get_mu()

    def get_cov(self):
        return np.array([item.cov for item in self.record_list])

    def get_cov_indicator(self):
        """
        cov实际上是多维方阵，这里使用取其行列式的值作为一维指示器，此处仅为了方便图像渲染，用以显示cov是否收敛，无物理方面的意义
        :return:
        """
        cov = self.get_cov()
        return np.apply_along_axis(lambda item: np.linalg.det(item), 2, cov)

    def get_pi(self):
        return np.array([item.pi for item in self.record_list])


def gaussian(X, mu_k, cov_k):
    """
    计算高斯分布的概率密度
    :param X: N 组 d维向量
    :param mu_k: 对应的数学期望
    :param cov_k: 协方差矩阵
    :return: N个长度的数值数组
    """
    d = X.shape[1]
    inv_cov = np.linalg.inv(cov_k)
    det_cov = np.linalg.det(cov_k)

    # 这里为计算 exp(-1 / 2 * (x - mu_k).T * cov_k ^ (-1) * (x - mu_k))
    def calculate_exp(x):
        temp_mu = x - mu_k
        return np.exp(-0.5 * np.matmul(np.matmul(temp_mu, inv_cov), temp_mu.T))

    # 使用apply_along_axis函数，对每一行进行计算
    temp = np.apply_along_axis(func1d=calculate_exp, axis=1, arr=X)

    # 高斯分布整体为 1/(2*pi)^(d/2) * 1/|cov_k|^(1/2) * exp(-1/2 * (x-mu_k).T * cov_k^(-1) * (x-mu_k))
    return 1 / (np.power(2 * np.pi, d / 2) * np.power(det_cov, 0.5)) * temp


def gmm_em(X, K, iteration, initial_mu=None):
    """
    GMM EM算法
    :param X: N 组 d维向量
    :param K: 高斯分布数量
    :param iteration: 目标迭代次数
    :param initial_mu: 给定值，初始的期望
    :return:
    """
    N, D = X.shape
    # Init
    P = np.ones((K, 1)) / K  # 初始化对应高斯分布的概率，开始将每个高斯分布的先验概率设置成相等的
    mu = np.random.rand(K, D) if initial_mu is None else initial_mu  # 初始化均值向量，使用随机向量生成，若未给定初始向量，则使用随机向量
    cov = np.array([np.eye(D)] * K)  # 初始化协方差矩阵，使用单位矩阵生成

    omega = np.zeros((N, K))  # 由算法推导的隐变量z的后验概率

    all_record = AllRecord(initial_mu=mu.copy())  # 记录每次迭代的结果, 用于绘制图像

    for i in range(iteration):
        # E-Step
        p = np.zeros((N, K))  # p为每个样本属于每个高斯分布的概率，因为有N个样本，K个高斯分布，所以p的维度为N*K
        for k in range(K):
            p[:, k] = P[k] * gaussian(X, mu[k], cov[k])  # 这里使用了之前定义的高斯分布函数，结合numpy的广播机制，计算出每个样本属于高斯分布k的概率
        sum_p = np.sum(p, axis=1)  # 求和，此处可得到N维向量，每个元素为该样本属于所有高斯分布的概率之和
        omega = p / sum_p[:, None]  # 计算隐变量z的后验概率，使用numpy的广播机制，将N维向量转换成N*K维矩阵

        # M-Step
        sum_omega = np.sum(omega, axis=0)  # 求和，此处可得到K维向量，每个元素为该高斯分布对应的所有样本的后验概率之和。该
        P = sum_omega / N  # alpha_k = sum(omega_k) / N
        for k in range(K):
            omega_x = np.multiply(X, omega[:, [k]])  # 最后得到的N*D维矩阵，每一行为对应样本的omega_k*X
            mu[k] = np.sum(omega_x, axis=0) / sum_omega[k]  # mu[k]  = sum(omega_k*X) / sum(omega_k) D维向量

            X_mu_k = np.subtract(X, mu[k])  # (X - mu_k) : [N*D] - [D] = [N*D]
            omega_X_mu_k = np.multiply(omega[:, [k]], X_mu_k)  # omega(X-mu_k) : [N*D]
            cov[k] = np.dot(np.transpose(omega_X_mu_k), X_mu_k) / sum_omega[
                k]  # sum(omega_i * (X_i-mu_k).T*(X_i-mu_k))  [D*D]
        all_record.record_list.append(Record(pi=P.copy(), mu=mu.copy(), cov=cov.copy()))
        if i % (iteration / 20) == 0:  # 这里相当于每执行iteration次数的5%便打印一下当前期望的进度
            print(f'当前遍历了{i + 1}次， 期望均值为{all_record.record_list[-1].mu}')

    return omega, P, mu, cov, all_record
