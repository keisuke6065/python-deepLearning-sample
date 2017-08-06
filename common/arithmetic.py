import numpy as np


# ソフトマックス関数
def softMax(np_array):
    # overflow対策
    np_max = np.max(np_array)
    np_exp = np.exp(np_array - np_max)
    np_sum = np.sum(np_exp)
    return np_exp / np_sum


# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ロス関数
# 二乗和誤差
def meanSquared(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# クロスエントロピー
def crossEntropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size


# 数値微分
def numericalDiff(func, x):
    # 0.0001
    h = 10e-4
    return (func(x + h) - func(x - h)) / (2 * h)


# 偏微分
def numericalGradient(func, x):
    # 0.0001
    h = 1e-4
    grad = np.zeros_like(x)

    for index in range(x.size):
        temp_val = x[index]
        # f(x+h)
        x[index] = float(temp_val) + h
        func1 = func(x)

        # f(x-h)
        x[index] = temp_val - h
        func2 = func(x)

        grad[index] = (func1 - func2) / (2 * h)
        x[index] = temp_val
    return grad
