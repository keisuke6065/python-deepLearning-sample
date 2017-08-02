import numpy as np

a = np.array([1.0, 2.0, 4.0])

exp_a = np.exp(a)
print(exp_a)
np_sum_a = np.sum(exp_a)
print(np_sum_a)

y = exp_a / np_sum_a
print(y)


def softMax(np_array):
    # overflow対策
    np_max = np.max(np_array)
    np_exp = np.exp(np_array - np_max)
    np_sum = np.sum(np_exp)
    return np_exp / np_sum


soft_max = softMax(np.array([3.0, 6.0, 9.0]))
soft_max_num = softMax(np.array([1000, 980, 960]))
print(soft_max)
print(soft_max_num)
print(np.sum(soft_max_num))