import numpy as np
import matplotlib.pyplot as plt
from common.arithmetic import numericalGradient


def function_2(X):
    if X.ndim == 1:
        return np.sum(X ** 2)
    else:
        return np.sum(X ** 2, axis=1)


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numericalGradient(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = numericalGradient(f, x)

        return grad


x = np.arange(-2, 2.5, 0.25)
y = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
grad = numerical_gradient(function_2, np.array([X, Y]))
plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1], angles='xy', color='#666666')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()
