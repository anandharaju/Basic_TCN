import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pickle
import os


def train_test_split(data, label, val_size, seed):
    idx = np.arange(len(data))
    np.random.seed(seed)
    np.random.shuffle(idx)
    split = int(len(data)*val_size)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    return x_train, x_test, y_train, y_test


def data_generator(data, labels, max_len, batch_size, shuffle):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)]
    while True:
        for i in batches:
            xx = preprocess(data[i], max_len)
            print(len(xx[0]))
            xx = xx.reshape(-1, len(xx[0]), 1)
            xx = xx.astype('float32')
            xx /= 255
            yy = labels[i]
            yy = np.expand_dims(yy, axis=2)
            yield (xx, yy)


def preprocess(file_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    corpus = []
    for fn in file_list:
        if not os.path.isfile(fn):
            print(fn, 'not exist')
        else:
            with open(fn, 'rb') as f:
                fjson = pickle.load(f)
                data = fjson["image_256w"]
                data = data.flatten()
                corpus.append(data)

    len_list = [max_len for doc in corpus]
    seq = pad_sequences(corpus, maxlen=max_len, truncating='post')  # , padding='post'
    return np.array(corpus)


if __name__ == '__main__':
    print(data_generator())
