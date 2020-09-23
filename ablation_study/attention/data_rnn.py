# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import h5py
import numpy as np
from keras.models import Model



def loaddata(filename):
    print('Start loading:',filename)
    print('    read file')
    file = h5py.File(filename, 'r')
    print('    x')
    train_x = file['x'][:]
    print('    y')
    train_y = file['y'][:]
    print('    valid_x')
    valid_x = file['valid_x'][:]
    print('    valid_y')
    valid_y = file['valid_y'][:]
    print('    test_x')
    test_x = file['test_x'][:]
    print('    test_y')
    test_y = file['test_y'][:]
    print('    close file')
    file.close()
    print('finish loading:',filename)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def load_NTU(case):

    if case == 0:
        filename = os.path.join('./data/ntu', 'NTU_CS.h5')
    else:
        filename = os.path.join('./data/ntu', 'NTU_CV.h5')

    train_x, train_y, valid_x, valid_y, test_x, test_y = loaddata(filename)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def get_data(dataset, case):

    if dataset == 'NTU':
        print('[data_rnn.py][get_data] Dataset: NTU, case:',case)
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_NTU(case)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def Downsample(train_x, valid_x, test_x, ratio):
    train_x = train_x[:,0::ratio,:]
    valid_x = valid_x[:,0::ratio,:]
    test_x = test_x[:, 0::ratio, :]
    return train_x, valid_x, test_x

def get_cases(dataset):
    if dataset == 'NTU':
        cases = 2

    return cases


def get_activation(model,test_x, test_y, path,  VA = 11 , par = 10):

    intermediat_layer_model = Model(inputs= model.input, outputs=model.layers[-2].output)
    score = intermediat_layer_model.predict(test_x, batch_size = 256)

    pred = softmax(score)
    pred_label = np.argmax(pred, axis= -1)
    label = np.argmax(test_y, axis= -1)

    total = ((label-pred_label)==0).sum()

    print("\nModel Accuracy:%.2f" % (float(total) / len(label)*100))
    np.savetxt(path, score, fmt = '%f')

    return (float(total)/len(label)*100)

def softmax(data):
    e = np.exp(data - np.amax(data, axis=-1, keepdims=True))
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s

