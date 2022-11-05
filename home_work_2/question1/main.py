from os import path

import numpy as np

from generate_data import GenerateData
from plot_image import plot_data, plot_record, get_path
from gmm_em import gmm_em


def calculate_order(predict_mu, default_mu):
    """
    计算真实的均值矢量与预测的均值矢量的对应关系
    :param predict_mu: 预测均值矢量
    :param default_mu: 真实均值矢量
    :return: 对应顺序
    """

    def distance(pre_mu):
        return np.apply_along_axis(lambda def_mu_item: np.dot(def_mu_item - pre_mu, def_mu_item - pre_mu), 1,
                                   default_mu)

    return np.argmin(np.apply_along_axis(distance, 1, predict_mu), axis=1)


data = GenerateData(load_data=True)


def question_1_2():
    # 绘制数据散点图(第一问)
    plot_data(data.result, data.result_target)
    # 给出估计(第二问)
    omega, P, mu, cov, record = gmm_em(data.result, K=len(data.all_mean), iteration=1000)
    order = calculate_order(mu, data.all_mean)
    P = P[order.tolist()]
    mu = mu[order.tolist(), :]
    cov = cov[order.tolist(), :]
    print(f'Order     : {order}')
    print(f'mu_initial: {record.initial_mu[order.tolist(), :]}')
    print(f'P_original: {data.P}')
    print(f'P_predict : {P}')
    print(f'P_error   : {np.abs(data.P - P)}')

    for i in range(len(data.all_mean)):
        print(f'mu_{i + 1}_original : {data.all_mean[i]}')
        print(f'mu_{i + 1}_predict  : {mu[i]}')
        print(f'mu_{i + 1}_error    : {np.abs(data.all_mean[i] - mu[i])}')

    for i in range(len(cov)):
        print(f'cov_{i + 1}_original : {data.COV[i]}')
        print(f'cov_{i + 1}_predict  : {cov[i]}')
        print(f'cov_{i + 1}_error    : {np.abs(data.COV[i] - cov[i])}')

    plot_record(record, data.get_target_record(), order)
    try:
        # 这里尝试引入生成实验数据报告的文件，若该文件不存在则不会继续生成
        from generate_experiment_result import write_experiment_result
        write_experiment_result(record.initial_mu[order.tolist(), :], P, mu, cov)
    except ImportError:
        # 在提交的作业中不会生成对应实验报告文件，此处直接放弃对错误进行处理
        pass


if __name__ == '__main__':
    question_1_2()  # 作业第一、二问
