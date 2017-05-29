from __future__ import print_function
from preprocessing import *
import numpy as np
np.random.seed(2333)  # for reproducibility

import os
#import cPickle as pickle
import time
import numpy as np
import h5py

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from STResNet import stresnet
import rmse as rmse

def build_model():
	model = stresnet(c_conf=(time_size, nb_channel, 52, 52), nb_residual_unit = 12)

	adam = Adam(lr=lr)
	model.compile(loss='mse', optimizer=adam, metrics=[rmse.rmse])

	model.summary()

	return model


if __name__ == '__main__':

	# set traing parameters
	nb_epoch = 500  # number of epoch at training stage
	nb_epoch_cont = 100  # number of epoch at training (cont) stage
	batch_size = 32  # batch size
	lr = 0.0002  # learning rate
	nb_residual_unit = 12  # number of residual units

	nb_channel = 1  # there are two types of flows: inflow and outflow
	map_height, map_width = 52, 52  # grid size
	path_result = 'RET'
	path_model = 'MODEL'

	# parameter initialization
	train_size = 2000
	test_size = 100
	time_size = 30

	if os.path.isdir(path_result) is False:
		os.mkdir(path_result)
	if os.path.isdir(path_model) is False:
		os.mkdir(path_model)

	# load_data
	print("loading data...")
	H = load_data('data.mat')
	
	# generate training data & test data
	print("generate data from data.mat")
	Data = H[:train_size + test_size, ]

	mmn = MinMaxNormalization()
	mmn.fit(Data)
	X_train, y_train = init_data(Data, train_size, time_size)
	X_test, y_test = init_data(Data[train_size:, ], test_size, time_size)
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	# build model
	model = build_model()

	ts = time.time()
	hyperparams_name = 'train_size{}.test_size{}.time_size{}.resunit{}.lr{}'.format(
        train_size, test_size, time_size, nb_residual_unit, lr)
	fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

	early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
	model_checkpoint = ModelCheckpoint(
    	fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')

	print("\nelapsed time (compiling model): %.3f seconds\n" %
        (time.time() - ts))

	print('=' * 10)

	print("load weight...")

	model.load_weights(os.path.join('MODEL', 'train_size1000.test_size100.time_size30.resunit12.lr0.0002.final.best.h5'))
	print('=' * 10)

	print("training model...")
	ts = time.time()

	history = model.fit(X_train, y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
                        model_checkpoint])
	print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

	json_string = model.to_json()
	open('model_json', 'w').write(json_string)
	#model = model_from_json(json_string)
	model.save_weights(os.path.join('MODEL', '{}.final.best.h5'.format(hyperparams_name)), overwrite=True)

	print('=' * 10)
	ts = time.time()
	score = model.evaluate(
    	X_test, y_test, batch_size=y_test.shape[0], verbose=0)
	print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
        (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
	print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))



