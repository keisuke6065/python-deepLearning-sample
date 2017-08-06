import numpy as np
import matplotlib.pyplot as plt
from common.arithmetic import meanSquared, crossEntropy, numericalDiff

y = np.array([0.1, 0.05, 0.6, 0.05, 0.2])
t = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

print(meanSquared(y, t))

print(crossEntropy(y, t))

print(np.shape(y))


def sampleFunction(X):
    return 0.01 * X ** 2 + 0.1 * X


print(numericalDiff(sampleFunction, 5))
print(numericalDiff(sampleFunction, 10))
print(numericalDiff(sampleFunction, 15))
print(numericalDiff(sampleFunction, 20))
x = np.arange(0.0, 20.0, 0.1)
y = sampleFunction(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()


