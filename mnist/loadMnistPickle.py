import os.path
import gzip
import pickle
import os
import numpy as np

dataSet_dir = os.path.dirname(os.path.abspath(__file__))
print(dataSet_dir)
save_file = dataSet_dir + "/pickle/mnist.pkl"

train_img = 'train_img'
train_label = 'train_label'
test_img = 'test_img'
test_label = 'test_label'

sampleDataFile = {
    train_img: 'train-images-idx3-ubyte.gz',
    train_label: 'train-labels-idx1-ubyte.gz',
    test_img: 't10k-images-idx3-ubyte.gz',
    test_label: 't10k-labels-idx1-ubyte.gz'
}

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def convert_numpy():
    dataSet = {train_img: load_img(sampleDataFile.get(train_img)),
               train_label: load_label(sampleDataFile.get(train_label)),
               test_img: load_img(sampleDataFile.get(test_img)),
               test_label: load_label(sampleDataFile.get(test_label))}

    return dataSet


def load_img(file_name):
    file_path = dataSet_dir + '/../data/gz/' + file_name
    print('converting image' + file_name)
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def load_label(file_name):
    file_path = dataSet_dir + '/../data/gz/' + file_name
    print('convertin label' + file_name)
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def init_mnsit():
    dataSet = convert_numpy()
    print("Creating pickle file")
    with open(save_file, 'wb') as f:
        pickle.dump(dataSet, f, -1)
    print("done")


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnsit()
    with open(save_file, 'rb') as f:
        dataSet = pickle.load(f)

    if normalize:
        for key in (train_img, test_img):
            dataSet[key] = dataSet[key].astype(np.float32)
            dataSet[key] /= 255.0

    if not flatten:
        for key in (train_img, test_img):
            dataSet[key] = dataSet[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        dataSet[train_label] = change_one_hot_label(dataSet[train_label])
        dataSet[test_label] = change_one_hot_label(dataSet[test_label])

    return (dataSet[train_img], dataSet[train_label]), (dataSet[test_img], dataSet[test_label])


if __name__ == '__main__':
    init_mnsit()
