from __future__ import print_function
from preprocessing import *
import numpy as np
np.random.seed(2333)  # for reproducibility

import os
import time
import numpy as np
import h5py
import pickle as pickle
import sys

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from STResNet import stresnet
import metrics as metrics
from STCNN import *

def build_model():
	if ml == 'resnet':
		model = stresnet(c_conf=(time_size, nb_channel, 52, 52), nb_residual_unit = nb_residual_unit, filter=ft, filter_size=ftsz)
	if ml == 'cnn':
		model = seqCNN(n_flow=1, seq_len=time_size, map_height=52, map_width=52, filter=ft)

	adam = Adam(lr=lr)
	model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])

	model.summary()

	return model


if __name__ == '__main__':

	# set traing parameters
	nb_epoch = 100  # number of epoch at training stage
	nb_epoch_cont = 100  # number of epoch at training (cont) stage
	batch_size = 32  # batch size
	lr = 0.00003  # learning rate
	nb_residual_unit = int(sys.argv[4])  # number of residual units

	nb_channel = 1  # there are two types of flows: inflow and outflow
	map_height, map_width = 52, 52  # grid size
	path_result = 'RET'
	path_model = 'MODEL'

	# parameter initialization
	train_size = int(sys.argv[1])
	test_size = int(sys.argv[2])
	time_size = int(sys.argv[3])

	# start index
	start = 500

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
	#Data = H[start : start + train_size + test_size, ]

	mmn = MinMaxNormalization()
	Data = mmn.fit_transform(Data)
	X_train, y_train = init_data(Data, train_size, time_size)
	X_test, y_test = init_data(Data[train_size:, ], test_size, time_size)
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

	# build model
	ml = sys.argv[5]
	ft = int(sys.argv[6])
	ftsz = int(sys.argv[7])
	model = build_model()
	
	ts = time.time()
	if ml == 'resnet':
		hyperparams_name = 'Be-train_size{}.test_size{}.time_size{}.resunit{}.lr{}.filter{}'.format(
        	train_size, test_size, time_size, nb_residual_unit, lr, ft)
	if ml == 'cnn':
		hyperparams_name = 'seqCNN.train_size{}.test_size{}.time_size{}.lr{}'.format(
    	    train_size, test_size, time_size, lr)
	
	fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
	
	early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
	model_checkpoint = ModelCheckpoint(
    	fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

	print("\nelapsed time (compiling model): %.3f seconds\n" %
        (time.time() - ts))

	print("training model...")
	ts = time.time()
	history = model.fit(X_train, y_train, nb_epoch=nb_epoch, verbose=1,
						batch_size=batch_size, validation_split=0.1,
						callbacks=[early_stopping, model_checkpoint])
	model.save_weights(os.path.join(path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
	with open(os.path.join(path_result, "{}.history.pkl".format(hyperparams_name)), "wb") as f:
		pickle.dump(history.history, f)
	print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

	# save model_json
	json_string = model.to_json()
	open('model_json', 'w').write(json_string)
	#model = model_from_json(json_string)
	

	print('=' * 10)
	print('evaluating using the model that has the best loss on the valid set')
	ts = time.time()
	model.load_weights(fname_param)
	score = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
	print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
		  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
	score = model.evaluate(
		X_test, y_test, batch_size=y_test.shape[0], verbose=0)
	print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
		  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
	print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

	print('=' * 10)
	print("training model (cont)...")
	ts = time.time()
	fname_param = os.path.join(
		'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
	model_checkpoint = ModelCheckpoint(
		fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
	history = model.fit(X_train, y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
		model_checkpoint])
	pickle.dump((history.history), open(os.path.join(
		path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
	model.save_weights(os.path.join(
		'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
	print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts))

	print('=' * 10)
	print('evaluating using the final model')
	score = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
	print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
		  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
	ts = time.time()
	score = model.evaluate(
		X_test, y_test, batch_size=batch_size, verbose=0)
	print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
		  (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
	print("\nelapsed time (eval cont): %.3f seconds\n" % (time.time() - ts))


