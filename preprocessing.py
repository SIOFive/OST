from __future__ import print_function

import numpy as np
import scipy.io as sio
np.random.seed(2333)  # for reproducibility

def load_data(filename):
	'''
	import data.mat
	where H is an 10000 * 2704 matrix
	10000 is the time step
	2704 = 52 * 52 (the grid size, height = width = 52)
    '''
	data = sio.loadmat(filename)
	H = data['Data']
	return H


def init_data(data, data_size, time_size):
    '''
    # generate (X,y) from data.mat
    
    :param data:            data from data.mat
    :param data_size:       the number of time steps 
    :param time_size:       the time size of each X
    :return:                X & y
    '''
    size = data_size - time_size
    X = []
    y = []
    for i in range(0, size):
        X.append(data[i: i + time_size, :].reshape(time_size, 2704))
        X[i] = X[i].reshape(time_size * 1, 52, 52)

        y.append(data[i + time_size, :])
        y[i] = y[i].reshape(1, 52, 52)

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='float32')
    return X, y

'''
def MinMaxNormalization(data):p
    mx = np.max(data)
    mn = np.min(data)
    data = (data - mn) / (mx - mn)
    data = data * 2 - 1

    return data, mx, mn
'''

class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X