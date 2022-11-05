from typing import overload

import numpy as np

from image_plot import ImagePlot


def normalize_M(X):
    return np.max(X, axis=0) - np.min(X, axis=0)


def calculate_euclidean_distance(X_train, normalize_m, x_predict):
    """
    计算欧式距离
    :param X_train: N x 2维训练集
    :param normalize_m: 归一化分母(这里不用计算，直接传入)
    :param x_predict: 需要预测的点
    :return: 预测点对于每个训练点的欧式距离
    """

    def temp(x):  # 这里定义一个临时函数，是因为匿名表达式无法快速实现。
        temp_data = (x - x_predict) / normalize_m
        return np.sqrt(np.dot(temp_data, temp_data))

    return np.apply_along_axis(temp, 1, X_train)


class KnnClassifier(ImagePlot):
    def __init__(self, train_data, train_labels, k, normalize_m=None):
        self.train_data = train_data
        self.train_labels = train_labels
        self.k = k
        self.normalize_m = normalize_M(train_data) if normalize_m is None else normalize_m

    def get_k_large_index(self, test_data):
        """
        获取前k个最大的欧式距离对应的索引
        :param test_data: 需要预测的点
        :return: 前k个最大的欧式距离对应的索引
        """
        euclidean_distance = calculate_euclidean_distance(self.train_data, self.normalize_m, test_data)
        return np.argsort(euclidean_distance)[:self.k]

    def predict_single(self, test_data):
        """
        预测单个点的分类
        :param test_data: 需要预测的点
        :return: 预测结果
        """
        k_large_index = self.get_k_large_index(test_data)
        # bincount 用于统计数组中每个值出现的次数，假设数组中的值是[1, 1, 1, 2],那么计算结果则为 [0, 3, 1], 0表示数组中没有0，1表示数组中有3个1，2表示数组中有1个2
        # argmax 用于获取数组中最大值的索引
        return np.argmax(np.bincount(self.train_labels[k_large_index]))

    def predict(self, Predict_Data):
        """
        预测给定多个点的分类
        :param Predict_Data: 需要预测的点
        :return: 预测结果
        """
        # apply_along_axis 等价于 for 循环
        return np.apply_along_axis(self.predict_single, 1, Predict_Data)
