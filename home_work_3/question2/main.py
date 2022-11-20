import numpy as np

from generate_data import GenerateData
import plot_image


# 问题二
def least_squares(X, Y):
    """
    通用最小二乘法
    :param X: 输入数据集
    :param Y: 输出数据集
    :return: 返回最小二乘法的系数
    """
    X_T = X.T

    W = np.linalg.inv(np.matmul(X_T, X))
    W = np.matmul(W, X_T)
    W = np.matmul(W, Y)

    return W


def gradient_descent(X, Y, iterations=1000, alpha=0.001):
    """
    梯度下降法
    :param X: 输入数据集
    :param Y: 输出数据集
    :param iterations: 迭代次数
    :param alpha: 学习率
    :return: 返回梯度下降法的系数
    """
    W = np.zeros((X.shape[1], 1))
    history = []
    for i in range(iterations):
        W = W - alpha * np.matmul(X.T, np.matmul(X, W) - Y)
        history.append(compute_E(X, Y, W)[0, 0])
        if i % (iterations / 100) == 0:  # 每迭代百分之一，输出一次进度
            print('iteration: %d, E: %.4f' % (i, history[-1]))

    return W, history


def compute_E(X, Y, W):
    """
    计算误差
    :param X: 输入数据集
    :param Y: 输出数据集
    :param W: 系数
    :return: 返回误差
    """
    temp = np.matmul(X, W) - Y
    return np.matmul(temp.T, temp) / 2


def compute_root_mean_square_error(X, Y, W):
    """
    计算均方根误差
    :param X: 输入数据集
    :param Y: 输出数据集
    :param W: 系数
    :return: 返回均方根误差
    """
    temp = np.matmul(X, W) - Y
    return np.sqrt(np.matmul(temp.T, temp) / X.shape[0])


if __name__ == '__main__':
    # 问题一
    # 生成数据集
    gd = GenerateData(load_data=True)
    # 绘制原始数据和函数散点
    plot_image.plot_data(gd.target_x, gd.target_y, gd.function_x)
    # 处理数据集并转为矩阵
    X = np.array([np.ones(gd.size), gd.target_x]).T
    Y = gd.target_y.reshape(gd.size, 1)

    # 问题三

    try:
        # 这里是用于生成最终pdf的代码，在正式运行流程中无影响
        import generate_tex

        generate_tex.generate_data_tex(Y)
    except ImportError as e:
        pass

    # 计算最小二乘法的系数
    W_L = least_squares(X, Y)
    W_G, history = gradient_descent(X, Y, iterations=5000, alpha=0.001)
    # 绘制梯度下降法的误差变化曲线
    plot_image.plot_gradient_history(history)

    try:
        # 这里是用于生成最终pdf的代码，在正式运行流程中无影响
        import generate_tex

        generate_tex.generate_param_tex(W_L, X, Y)
    except ImportError as e:
        pass

    print('W_L: ', W_L, 'W_G: ', W_G)
    plot_image.plot_predict_line(W_L, gd.target_x, gd.target_y, gd.function_x)

    print('1次项 均方误差为: %.4f' % (compute_E(X, Y, W_L)))
    W_List = [W_L]
    # 第五问
    # 此处为函数的真实值
    domain_x = np.arange(-5, 5, 0.1)
    # 计算 -5 到 5 之间的真实函数值
    domain_Y = gd.function_x(domain_x).reshape(domain_x.shape[0], 1)

    # 这里使用数组是为了方便后续的矩阵转换，使其不用在提高幂后的下次循环中再重新计算已有数据
    data_array = [np.ones(gd.size), gd.target_x]
    real_data_array = [np.ones(domain_x.shape[0]), domain_x]

    Y = gd.target_y.reshape(gd.size, 1)

    min_real_error = 1000000
    min_poly_index = 0
    min_poly_W_L = None
    for i in range(2, gd.size - 1):
        # 添加幂数据
        data_array.append(gd.target_x ** i)
        real_data_array.append(domain_x ** i)
        # 转换为矩阵
        X = np.array(data_array).T
        # 转换真实X的矩阵
        domain_X = np.array(real_data_array).T
        # 计算最小二乘法的系数
        W_L = least_squares(X, Y)
        # 绘制拟合曲线
        plot_image.plot_predict_line(W_L, gd.target_x, gd.target_y, gd.function_x)
        # 计算均方误差
        error = compute_root_mean_square_error(X, Y, W_L)
        # 计算真实函数的均方误差
        real_error = compute_root_mean_square_error(domain_X, domain_Y, W_L)
        print('%d次项 均方根误差为: %.4f, 真实函数均方根误差为: %.4f' % (i, error, real_error))
        # 记录最小的真实函数均方误差
        if real_error < min_real_error:
            min_real_error = real_error
            min_poly_index = i
            min_poly_W_L = W_L

    print('最小的真实函数均均方根误差为: %.4f, 对应的多项式为: %d次项' % (min_real_error, min_poly_index))

    try:
        # 这里是用于生成最终pdf的代码，在正式运行流程中无影响
        import generate_tex

        generate_tex.generate_poly_tex(min_poly_W_L)
    except ImportError as e:
        pass
