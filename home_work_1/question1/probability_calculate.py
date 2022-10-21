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
    :return: 正态分布概率密度函数数值
    """
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    x_mean = x - mean
    temp = np.matmul(x_mean, cov_inv)
    temp = np.matmul(temp, x_mean.T)
    return 1 / (np.power(2 * np.pi, len(x) / 2) * np.power(cov_det, 1 / 2)) * np.exp(
        -1 / 2 * temp)


class ProbabilityCalculate:
    """
    进行决策率计算的父类函数，其具体实现由各个子类去完成
    """
    name = 'Default'  # 记录当前预测函数类的中文名称，用于在绘图时显示
    en_name = 'Default'  # 文件名称中的英文名称
    predict_result = None
    check_max = True

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
        m_all = self.data.default_mean
        for k in range(len(m_all)):
            temp = self.calculate_g_k(self.data.real_data_array[i][j], i, k)
            if result is None or (result < temp if self.check_max else result > temp):
                result_index = k
                result = temp
        return result_index + 1

    def predict(self):
        final_result = []
        for i in range(len(self.data.real_data_array)):
            print(f'正在计算数据分布X{i + 1}使用“{self.name}”的预测值')
            X = self.data.real_data_array[i]
            y = self.data.target_value_array[i]
            size = self.data.default_size
            error_count = 0
            result_temp = []
            for _ in range(len(self.data.default_mean)):
                result_temp.append([])
            result_error_temp = []
            for j in range(size):
                predict = self.calculate_predict(i, j)
                if predict != y[j]:
                    error_count += 1
                    result_error_temp.append([X[j][0], X[j][1]])
                else:
                    result_temp[predict - 1].append([X[j][0], X[j][1]])
            result = []
            for item in result_temp:
                result.append(np.array(item))

            result_error = np.array(result_error_temp)
            final_result.append((result, result_error))
            print(f'数据分布X{i + 1}使用“{self.name}”的预测值计算完毕')
        self.predict_result = final_result

    def print_error_rate(self):
        for i in range(len(self.predict_result)):
            print(f'数据分布X{i + 1}使用“{self.name}”的预测错误率为：'
                  f'{len(self.predict_result[i][1]) / self.data.default_size}')

    def plot_error_and_line(self):
        """
        绘制散点图包含失败以及数据分割线
        """
        for i in range(len(self.predict_result)):
            print(f'开始绘制“{self.name}”的X{i + 1}数据分布及散点图')
            file_name = f'X{i + 1}_data_{self.en_name}_predict_result.png'
            plot_image.plot_data_line(self.predict_result[i][0], self.predict_result[i][1],
                                      f'X{i + 1} {self.name} 预测结果',
                                      file_name,
                                      lambda x, k: self.calculate_g_k(x, i, k),  # 这里使用lambda表达式是为了增加接下来的代码复用性
                                      self.check_max
                                      )
            print(f'“{self.name}”的X{i + 1}数据分布及散点图绘制完毕，图片保存在当前目录下的 {file_name}。')


class LikelihoodProbability(ProbabilityCalculate):
    """
    似然概率的计算类
    """
    name = '似然率决策规则'
    en_name = 'likelihood'

    def calculate_g_k(self, x, i, j) -> float:
        return calculate_normal_distribution(x, self.data.default_mean[j], self.data.default_cov) * \
               self.data.probabilities_array[i][j]


class BayesProbability(ProbabilityCalculate):
    """
    贝叶斯概率的计算类
    """
    name = '贝叶斯风险决策规则'
    en_name = 'bayes'
    check_max = False
    C = [[0, 2, 3],
         [1, 0, 2.5],
         [1, 1, 0]]

    def calculate_bayes_probability(self, cov, x, m, p):
        temp = calculate_normal_distribution(x, m, cov)
        temp *= p * self.data.default_size
        return temp

    def calculate_g_k(self, x, i, j) -> float:
        temp = 0
        m_all = self.data.default_mean
        p_all = self.data.probabilities_array[i]
        for k in range(len(m_all)):
            temp += self.C[j][k] * self.calculate_bayes_probability(self.data.default_cov,
                                                                    x,
                                                                    m_all[k],
                                                                    p_all[k])
        return temp


class EuclidProbability(ProbabilityCalculate):
    """
    欧式距离的计算类
    """
    name = '最小欧几里得距离分类器'
    en_name = 'Euclid'

    def calculate_g_k(self, x, i, j) -> float:
        temp = x - self.data.default_mean[j]
        return - (1 / 2) * np.dot(temp, temp)
