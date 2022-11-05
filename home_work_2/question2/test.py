import numpy as np

if __name__ == '__main__':
    data = np.load('data/knn_100_accuracy.npy')
    print(data[0], data[49])