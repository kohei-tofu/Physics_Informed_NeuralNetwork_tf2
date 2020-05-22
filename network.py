 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


def get_linear(layer_list):
    #glorot_normal
    #glorot_uniform
    num_layers = len(layer_list) 
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layer_list[1], \
                                    input_shape = (layer_list[0],),\
                                    activation = 'tanh',\
                                    kernel_initializer='glorot_normal'))

    for l in range(1, num_layers - 2):
        model.add(tf.keras.layers.Dense(layer_list[l+1], \
                                        activation='tanh',\
                                        kernel_initializer='glorot_normal'))
    #tf.keras.layers.Activation('tanh'),
    model.add(tf.keras.layers.Dense(layer_list[-1], \
                                    activation=None,\
                                    kernel_initializer='glorot_normal'))
    
    model.summary()
    return model


def get_linear2(layer_list):
    num_layers = len(layer_list) 
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layer_list[1], \
                                    input_shape = (layer_list[0],),\
                                    activation = 'tanh',\
                                    kernel_initializer='glorot_normal'))

    for l in range(1, num_layers - 2):
        model.add(tf.keras.layers.Dense(layer_list[l+1], \
                                        activation='tanh',\
                                        kernel_initializer='glorot_normal'))
    #tf.keras.layers.Activation('tanh'),
    model.add(tf.keras.layers.Dense(layer_list[-1], \
                                    activation='tanh',\
                                    kernel_initializer='glorot_normal'))
    
    model.add(tf.keras.layers.Lambda(lambda x: x * 5.))
    model.summary()
    return model


