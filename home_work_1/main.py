import generate_data as gd
import plot_image as pi
import numpy as np
import probability_calculate as pc
# 这里使用类生成是为了便于接下来做重复实验
Data = gd.GenerateData(
    default_mean=[[1, 1], [4, 4], [8, 1]],  # 均值数组
    default_cov=np.matrix([[2, 0], [0, 2]]),  # 协方差矩阵
    default_size=1000,  # 生成数据的数量
    probabilities_array=[[0.333, 0.333, 0.334], [0.6, 0.3, 0.1]]  # 先验概率数组，数组中的每个元素为一个先验概率生成模型
)

pi.plot_data(Data.plot_data_array[0], None, 'X1', 'X1_data.png')
pi.plot_data(Data.plot_data_array[1], None, 'X2', 'X2_data.png')

if __name__ == '__main__':
    likelihood = pc.LikelihoodProbability(Data)
    likelihood.predict()
    likelihood.plot_error_and_line()