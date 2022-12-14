import generate_data as gd
import plot_image as pi
import probability_calculate as pc
# 使用三方库numpy进行矩阵等数据的计算
import numpy as np

# 这里使用类生成是为了便于接下来做重复实验
Data = gd.GenerateData(
    default_mean=np.array([[1, 1], [4, 4], [8, 1]]),  # 均值数组
    default_cov=np.matrix([[2, 0], [0, 2]]),  # 协方差矩阵
    default_size=1000,  # 生成数据的数量
    probabilities_array=[[0.333, 0.333, 0.334], [0.6, 0.3, 0.1]]  # 先验概率数组，数组中的每个元素为一个先验概率生成模型
)
# 第一小问的散点图绘制
# 绘制散点图
pi.plot_data(Data.plot_data_array[0], None, 'X1', 'X1_data.png')
pi.plot_data(Data.plot_data_array[1], None, 'X2', 'X2_data.png')

if __name__ == '__main__':
    test = [
        pc.LikelihoodProbability(Data),  # 似然决策规则 分类器
        pc.BayesProbability(Data),  # 贝叶斯 分类器
        pc.EuclidProbability(Data),  # 最小欧几里得距离分类器
    ]

    for item in test:
        item.predict()  # 执行推理过程
        item.plot_error_and_line()  # 进行绘制操作（绘制散点图，包含错误预估数据 以及 分类线）

    for item in test:
        item.print_error_rate()  # 打印当前类型预测错误率
