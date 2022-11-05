import os
import time
from datetime import datetime
from os import path

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from home_work_2.question2.knn_classifier import KnnClassifier
from home_work_2.question2.likelihood_classifier import LikelihoodClassifier


# 归一化分母
def normalize_M(X):
    return np.max(X, axis=0) - np.min(X, axis=0)


def compute_max_k_accuracy(max_k, X_train, y_train, X_test, y_test, load_accuracy_from_file=False):
    """
    计算从1至max_k的knn算法准确率
    :param max_k: 最大k值
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param load_accuracy_from_file: 是否读取本地文件的准确率(用以减少重复运算时间)
    """
    accuracy = np.zeros(max_k)
    file = path.join('data', f'knn_{max_k}_accuracy.npy')
    if load_accuracy_from_file and path.exists(file):
        accuracy = np.load(file)
    else:
        normalize_m = np.max(X_train, axis=0) - np.min(X_train, axis=0)
        begin_time = datetime.now()
        for k in range(1, max_k + 1):
            start_time = datetime.now()
            knn_classifier = KnnClassifier(X_train, y_train, k, normalize_m)
            y_predict = knn_classifier.predict(X_test)
            accuracy[k - 1] = calculate_accuracy(y_predict, y_test)
            print(f'k = {k: 2d}, accuracy = {accuracy[k - 1]: .6f}, 耗时: {datetime.now() - start_time}')

        print(f'总数量：{max_k}，总耗时：{datetime.now() - begin_time}')
        if not path.exists('data'):
            os.mkdir('data')
        np.save(file, accuracy)

    plt.plot(range(1, max_k + 1), accuracy)
    # plt.show()
    plt.xlabel('k')
    plt.ylabel('test accuracy')
    plt.savefig('knn_accuracy.png')
    plt.close()
    best_k = np.argmax(accuracy) + 1
    print(f'最佳k值为：{best_k}，准确率为：{accuracy[best_k - 1]: .6f}')
    return best_k


def calculate_accuracy(y_predict, y_test):
    """
    计算准确率
    :param y_predict: 预测结果
    :param y_test: 测试集标签
    :return: 准确率
    """
    return np.sum(y_predict == y_test) / len(y_test)


def calculate_cov(X):
    """
    计算协方差矩阵
    :param X: N x 2维训练集
    :return: 协方差矩阵
    """
    return np.cov(X, rowvar=False)


def calculate_mu(X):
    """
    计算均值
    :param X: N x 2维训练集
    :return: 均值
    """
    return np.sum(X, axis=0) / len(X)
    # return np.mean(X, axis=0)


if __name__ == '__main__':
    # 加载数据
    data = loadmat('HW#2.mat')
    c1 = data['c1']
    c2 = data['c2']
    c3 = data['c3']
    t1 = data['t1']
    t2 = data['t2']
    t3 = data['t3']
    # 合并训练集并将标签转换为0,1,2
    X_train = np.concatenate((c1, c2, c3), axis=0)
    y_train = np.concatenate((np.zeros(c1.shape[0]), np.ones(c2.shape[0]), np.ones(c3.shape[0]) * 2), axis=0).astype(
        np.int32)

    # 合并测试数据集并将标签转换为0,1,2
    X_test = np.concatenate((t1, t2, t3), axis=0)
    y_test = np.concatenate((np.zeros(t1.shape[0]), np.ones(t2.shape[0]), np.ones(t3.shape[0]) * 2), axis=0).astype(
        np.int32)

    max_k = 100  # 这里使用100个点来做测试，接下来可通过观察准确率变化趋势，可以得到是否继续使用更大的k值
    best_k = compute_max_k_accuracy(max_k, X_train, y_train, X_test, y_test, load_accuracy_from_file=True)

    # 似然估计
    mu1 = calculate_mu(c1)
    mu2 = calculate_mu(c2)
    mu3 = calculate_mu(c3)

    mu = np.array([mu1, mu2, mu3])

    cov1 = calculate_cov(c1)
    cov2 = calculate_cov(c2)
    cov3 = calculate_cov(c3)

    cov = np.array([cov1, cov2, cov3])
    # 似然分类器
    likelihood_classifier = LikelihoodClassifier(X_train, y_train, mu, cov)

    # 似然分类器预测
    y_predict = likelihood_classifier.predict(X_test)
    likelihood_accuracy = calculate_accuracy(y_predict, y_test)
    print(f'似然分类器准确率为{likelihood_accuracy}')
    # 绘制决策边界
    knn_classifier = KnnClassifier(X_train, y_train, best_k)

    likelihood_classifier.plot(X_test, y_test, 'likelihood_classifier.png')
    # 注意这里绘制knn分类器的决策边界，速度很慢，在上述compute_max_k_accuracy函数中可以看出，对于一组1500数据点的训练集，耗时为约为4.7s。
    # 绘制分类边界时，需要对每个点进行预测，此时使用的点阵为 1000 * 1000 = 1000000个点，因此预测耗时为
    # 1000000 / 1500 * 4.7 = 3133s = 52.2min = 0.87h, 此时可以考虑降低绘图边界。或者使用更优算法（如kd树）以优化计算训练集的时间。
    knn_classifier.plot(X_test, y_test, 'knn_decision_boundary.png')
