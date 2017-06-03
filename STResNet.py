from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Reshape
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=True):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(c_conf=(10, 1, 52, 52), nb_residual_unit = 3, filter=64):
    # conf = (len_seq, nb_channel, map_height, map_width)
    
    # main input
    main_inputs = []
    outputs = []
    #filter = 64
    if c_conf is not None:
        len_seq, nb_channel, map_height, map_width = c_conf
        input = Input(shape=(nb_channel * len_seq, map_height, map_width))
        main_inputs.append(input)
        # Conv1
        conv1 = Convolution2D(
            nb_filter=filter, nb_row=3, nb_col=3, border_mode="same")(input)
        # [nb_residual_unit] Residual Units
        residual_output = ResUnits(_residual_unit, nb_filter=filter,
                                   repetations=nb_residual_unit)(conv1)
        # Conv2
        activation = Activation('relu')(residual_output)
        conv2 = Convolution2D(
            nb_filter=nb_channel, nb_row=3, nb_col=3, border_mode="same")(activation)
        outputs.append(conv2)

    main_output = outputs[0]

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model