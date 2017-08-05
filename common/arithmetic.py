import numpy as np


def softMax(np_array):
    # overflow対策
    np_max = np.max(np_array)
    np_exp = np.exp(np_array - np_max)
    np_sum = np.sum(np_exp)
    return np_exp / np_sum


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
