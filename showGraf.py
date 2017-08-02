import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

grafX = np.arange(0, 8, 0.1)
grafY = np.sin(grafX)
grafY2 = np.cos(grafX)
plt.plot(grafX, grafY, label="sin")
plt.plot(grafX, grafY2, label="cos", linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('sin&cos')
plt.show()

# img = imread('./thumbs.jpeg')
# plt.imshow(img)
# plt.show()
