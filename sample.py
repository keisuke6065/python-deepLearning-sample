import numpy as np


x = np.array([1.0, 2.0, 3.0])
print(x)
print(x / 2)

xx = np.array([[1.0, 2.0], [3.0, 4.0]])
print(xx)
print(xx.shape)
print(xx.dtype)

yy = np.array([[5.0, 6.0], [7.0, 8.0]])
print(yy)
print(xx + yy)
print(xx * yy)

print(xx[0])
print(yy[0])

for row in xx:
    print(row)

print(xx.flatten())
print(xx > 2)

