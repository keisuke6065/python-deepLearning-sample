import numpy as np


# import sigmoid_sample as sigmoid


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.array([1.0, 0.5])
w1 = np.array([[1.0, 0.3, 0.5], [0.3, 0.4, 0.5]])
b1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, w1) + b1
Z1 = sigmoid(A1)
w2 = np.array([[0.3, 0.5], [0.8, 0.2], [0.4, 0.5]])
b2 = np.array([0.4, 0.5])
A2 = np.dot(Z1, w2) + b2
w3 = np.array([[0.3, 0.7], [0.5, 0.2]])
b3 = np.array([0.1, 0.8])
A3 = np.dot(A2, w3) + b3
print(A1)
print(A2)
print(sigmoid(A1))
print(sigmoid(A2))
# 三層の回帰問題
print(sigmoid(A3))
