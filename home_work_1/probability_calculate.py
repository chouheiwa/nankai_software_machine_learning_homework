"""
用于提供各种计算的类
"""
import plot_image
import numpy as np

from generate_data import GenerateData


def calculate_normal_distribution(x, mean, cov):
    """
    计算正态分布的概率密度函数
    :param x: 输入的数据
    :param mean: 均值
    :param cov: 协方差
    :return:
    """
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    x_mean = x - mean
    temp = np.matmul(x_mean, np.linalg.inv(cov))
    temp = np.matmul(temp, x_mean.T)
    return 1 / (np.power(2 * np.pi, len(x) / 2) * np.power(cov_det, 1 / 2)) * np.exp(
        -1 / 2 * temp)


class ProbabilityCalculate:
    """
    进行决策率计算的父类函数，其具体实现由各个子类去完成
    """
    name = 'Default'
    predict_result = None

    def __init__(self, data: GenerateData):
        self.data = data

    def calculate_g_k(self, x, i, j) -> float:
        """
        计算第i个数据分布Xi下，第j个数据的gk(X)
        :param x: 向量
        :param i: 第i个数据分布Xi
        :param j: 计算g_j(x)
        :return: g_j(x)
        """
        assert '需要被子类函数实现'
        return 0

    def calculate_predict(self, i: int, j: int) -> int:
        """
        计算第i个数据分布Xi下，第j个数据的预测值
        :param i: 第i个数据分布Xi
        :param j: 第j个数据
        :return: 预测分类
        """
        result = None
        result_index = 0
        m_all = self.data.default_mean[i]
        for k in range(len(m_all)):
            temp = self.calculate_g_k(self.data.real_data_array[i][j], i, k)
            if result is None or result < temp:
                result_index = i
                result = temp
        return result_index + 1

    def predict(self):
        final_result = []
        for i in range(len(self.data.real_data_array)):
            print(f'正在计算数据分布X{i + 1}的预测值')
            X = self.data.real_data_array[i]
            y = self.data.target_value_array[i]
            size = X.shape[0]
            error_count = 0
            result_temp = []
            for _ in range(len(self.data.default_mean[i])):
                result_temp.append([])
            result_error_temp = []
            for j in range(size):
                predict = self.calculate_predict(i, j)
                if predict != y[i]:
                    error_count += 1
                    result_error_temp.append([X[i][0], X[i][1]])
                else:
                    result_temp[predict - 1].append([X[i][0], X[i][1]])
            result = []
            for item in result_temp:
                result.append(np.array(item))

            result_error = np.array(result_error_temp)
            final_result.append((result, result_error))

        self.predict_result = final_result

    def plot_error_and_line(self):
        for i in range(len(self.predict_result)):
            plot_image.plot_data_line(self.predict_result[i][0], self.predict_result[i][1],
                                      f'X{i + 1} {self.name} predict',
                                      f'X{i + 1}_result.png', lambda x, k: self.calculate_g_k(x, i, k))


class LikelihoodProbability(ProbabilityCalculate):
    """
    似然概率的计算类
    """
    name = 'likelihood'

    def calculate_g_k(self, x, i, j) -> float:
        return calculate_normal_distribution(x, self.data.default_mean[i], self.data.default_cov) * \
               self.data.probabilities_array[i][j]


class BayesProbability(ProbabilityCalculate):
    """
    贝叶斯概率的计算类
    """
    name = 'bayes'
    C = [[0, 2, 3],
         [1, 0, 2.5],
         [1, 1, 0]]

    def calculate_bayes_probability(self, cov, x, m, p):
        temp = calculate_normal_distribution(x, m, cov)
        temp *= p / (1 / self.data.default_size)
        return temp

    def calculate_g_k(self, x, i, j) -> float:
        temp = 0
        m_all = self.data.default_mean[i]
        for j in range(len(m_all)):
            temp += self.C[i][j] * self.calculate_bayes_probability(self.data.default_cov, x, m_all[i],
                                                                    self.data.probabilities_array[i])
        return temp


class EuclidProbability(ProbabilityCalculate):
    """
    欧式距离的计算类
    """
    name = 'euclid'

    def calculate_g_k(self, x, i, j) -> float:
        temp = x - self.data.default_mean[i][j]
        return - (1 / 2) * np.dot(temp, temp)
