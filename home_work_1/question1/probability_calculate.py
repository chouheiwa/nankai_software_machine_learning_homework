"""
用于提供各种计算的类
"""
import plot_image
import numpy as np

from generate_data import GenerateData


def calculate_normal_distribution(x, mean, cov):
    """
    计算正态分布的概率密度函数-这里去除了常数项
    :param x: 输入的数据
    :param mean: 均值
    :param cov: 协方差
    :return: 正态分布概率密度函数数值
    """
    cov_inv = np.linalg.inv(cov)
    x_mean = x - mean
    temp = np.matmul(x_mean, cov_inv)
    temp = np.matmul(temp, x_mean.T)
    return np.exp(- 0.5 * temp)


def calculate_normal_distribution_with_x_y(x, y, mean, cov):
    cov_inv = np.linalg.inv(cov)
    x_mean = x - mean[0]
    y_mean = y - mean[1]
    temp = np.power(x_mean, 2) * cov_inv[0, 0] + np.power(y_mean, 2) * cov_inv[1, 1]
    return np.exp(-0.5 * temp)


class ProbabilityCalculate:
    """
    进行决策率计算的父类函数，其具体实现由各个子类去完成
    """
    name = 'Default'  # 记录当前预测函数类的中文名称，用于在绘图时显示
    en_name = 'Default'  # 文件名称中的英文名称
    predict_result = None
    check_max = True  # 用于判别规则中进行g_k是否为最大值

    def __init__(self, data: GenerateData):
        self.data = data

    def calculate_g_k_with(self, x1, x2, i, j):
        """
        用于供numpy加速计算时使用，相当于限制了二维数组，减少画图时的计算量
        :param x1: x1的值
        :param x2: x2的值
        :param i: 第i个数据分布Xi
        :param j: 第j个数据分布Xj
        :return: g_j(x)
        """
        assert '需要被子类函数实现'
        pass

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
        """
        对数据集进行预测
        """
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
        """
        打印对应数据集的错误率
        """
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
                                      lambda x, y, k: self.calculate_g_k_with(x, y, i, k),
                                      # 这里使用lambda表达式是为了增加接下来的代码复用性
                                      self.check_max
                                      )
            print(f'“{self.name}”的X{i + 1}数据分布及散点图绘制完毕，图片保存在当前目录下的 {file_name}。')


class LikelihoodProbability(ProbabilityCalculate):
    """
    似然概率的计算类
    """
    name = '似然率决策规则'
    en_name = 'likelihood'

    def calculate_g_k_with(self, x1, x2, i, j):
        return calculate_normal_distribution_with_x_y(x1,
                                                      x2,
                                                      self.data.default_mean[j],
                                                      self.data.default_cov) * \
               self.data.probabilities_array[i][j]

    def calculate_g_k(self, x, i, j) -> float:
        cov_inv = np.linalg.inv(self.data.default_cov)
        x_mean = x - self.data.default_mean[j]
        temp = np.matmul(x_mean, cov_inv)
        temp = np.matmul(temp, x_mean.T)

        return np.exp(-0.5 * temp) * self.data.probabilities_array[i][j]


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
        return calculate_normal_distribution(x, m, cov) * p

    def calculate_bayes_probability_with(self, cov, x1, x2, m, p):
        return calculate_normal_distribution_with_x_y(x1, x2, m, cov) * p

    def calculate_g_k_with(self, x1, x2, i, j):
        temp = None
        m_all = self.data.default_mean
        p_all = self.data.probabilities_array[i]
        for k in range(len(m_all)):
            current = self.C[j][k] * self.calculate_bayes_probability_with(self.data.default_cov,
                                                                           x1,
                                                                           x2,
                                                                           m_all[k],
                                                                           p_all[k])
            if temp is None:
                temp = current
            else:
                temp += current
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

    def calculate_g_k_with(self, x1, x2, i, j):
        x1_mean = x1 - self.data.default_mean[j][0]
        x2_mean = x2 - self.data.default_mean[j][1]
        return - (np.power(x1_mean, 2) + np.power(x2_mean, 2))

    def calculate_g_k(self, x, i, j) -> float:
        temp = x - self.data.default_mean[j]
        return - np.dot(temp, temp)
