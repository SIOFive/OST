from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding3D
from keras.layers.convolutional import Convolution2D, Convolution3D

def seqCNN(n_flow=1, seq_len=3, map_height=52, map_width=52, filter=64):
	model = Sequential()
	model.add(Convolution2D(filter, 3, 3, input_shape=(n_flow*seq_len, map_height, map_width), border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(filter, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(filter, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(n_flow, 3, 3, border_mode='same'))
	model.add(Activation('tanh'))
	return model

def seqCNN_BN(n_flow=1, seq_len=3, map_height=52, map_width=52):
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, input_shape=(n_flow*seq_len, map_height, map_width), border_mode='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Convolution2D(n_flow, 3, 3, border_mode='same'))
	model.add(Activation('tanh'))
	return model

def seq3DCNN(n_flow=1, seq_len=3, map_height=52, map_width=52):
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, input_shape=(n_flow*seq_len, map_height, map_width), border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(128, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(n_flow, 3, 3, border_mode='same'))
	model.add(Activation('tanh'))
	return model