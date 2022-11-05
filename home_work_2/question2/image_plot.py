from abc import abstractmethod, ABCMeta

import numpy as np
from matplotlib import pyplot as plt


class ImagePlot(metaclass=ABCMeta):
    """此类主要是为了使子类分类器可以直接继承获得绘图功能，而无需单独实现"""

    train_data = None
    train_label = None

    @abstractmethod
    def predict(self, test_data):
        pass

    def plot(self, test_data, test_label, file_name):
        fig = plt.figure(figsize=(12, 12))

        predict = self.predict(test_data)
        # 绘制散点图部分
        correct = np.concatenate(np.argwhere(predict == test_label), axis=0)
        wrong = np.concatenate(np.argwhere(predict != test_label), axis=0)
        # 绘制正确的散点图
        max_kind = np.max(test_label) + 1
        correct_data = test_data[correct]
        correct_label = test_label[correct]
        for i in range(max_kind):
            current_data = correct_data[correct_label == i]
            plt.plot(current_data[:, 0], current_data[:, 1], 'o', markersize=4, label=f'$i={i + 1}$')
        # 绘制错误的散点图
        wrong_data = test_data[wrong]
        plt.plot(wrong_data[:, 0], wrong_data[:, 1], 'o', markersize=4, label=f'wrong')
        plt.legend()

        # 绘制分界线
        x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
        y_min, y_max = test_data[:, 1].min() - 1, test_data[:, 1].max() + 1
        x = np.linspace(x_min, x_max, 1000)
        y = np.linspace(y_min, y_max, 1000)
        xx, yy = np.meshgrid(x, y)
        plot_data = np.c_[xx.ravel(), yy.ravel()]
        z = self.predict(plot_data)
        plt.contourf(xx, yy, z.reshape(xx.shape), alpha=0.2)
        plt.savefig(file_name)
        plt.close()
