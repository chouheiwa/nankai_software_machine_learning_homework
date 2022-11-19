from os import path

from main import compute_E
from plot_image import get_path


def generate_data_tex(Y):
    with open(get_path('data.template'), 'r') as f1:
        temp = f1.read().replace('${}', (' \\\\'.join(['%.4f' % y for y in Y])))
        with open(get_path(path.join('作业文档', 'data.tex')), 'w') as f:
            f.write(temp)


def generate_param_tex(W, X, Y):
    temp = r"\["
    temp += r"W = \begin{bmatrix}%.4f \\ %.4f \end{bmatrix} \]" % (W[0, 0], W[1, 0])
    temp += "\n"
    temp += r"\[\hat{y} = %.4f + %.4f x \]" % (W[0, 0], W[1, 0])
    temp += "\n"
    temp += r"最终预测均方误差为: \[E = %.4f\]" % compute_E(X, Y, W)[0, 0]
    with open(get_path(path.join('作业文档', 'param.tex')), 'w') as f:
        f.write(temp)


def generate_poly_tex(W_L):
    temp = r"\["
    temp += r"y = "

    length = W_L.shape[0]

    for i in range(length):
        if i == 0:
            temp += r"%f" % W_L[i, 0]
            continue
        if W_L[i, 0] > 0:
            temp += r" + "
        if W_L[i, 0] == 0:
            continue
        temp += r"%f x^%d" % (W_L[i, 0], i)

    temp += r"\]"
    with open(get_path(path.join('作业文档', 'poly.tex')), 'w') as f:
        f.write(temp)
