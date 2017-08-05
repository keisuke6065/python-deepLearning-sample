import numpy as np
from mnist.loadMnistPickle import load_mnist
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)

print(x_train.shape)
print(x_train)
print(t_train)
print(x_test)
print(t_test)


def show_image(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


train_img = x_train[0]
train_label = t_train[0]
print(train_label)
print(train_img.shape)
train_img = train_img.reshape(28, 28)
print(train_img.shape)

show_image(train_img)
