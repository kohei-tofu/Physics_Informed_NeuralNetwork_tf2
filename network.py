 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense




def get_linear1(layer_list):
    num_layers = len(layer_list) 
    model = tf.keras.Sequential()

    xavier_stddev = np.sqrt(2/(layer_list[0] + layer_list[1]))
    init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=xavier_stddev)
    #init = 'truncated_normal'
    model.add(tf.keras.layers.Dense(layer_list[1], \
                                    input_shape = (layer_list[0],),\
                                    activation = 'tanh',\
                                    kernel_initializer=init))

    for l in range(1, num_layers - 2):
        xavier_stddev = np.sqrt(2/(layer_list[l] + layer_list[l+1]))
        init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=xavier_stddev)
        model.add(tf.keras.layers.Dense(layer_list[l+1], \
                                        activation='tanh',\
                                        kernel_initializer=init))

    xavier_stddev = np.sqrt(2/(layer_list[-2] + layer_list[-1]))
    init = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=xavier_stddev)
    model.add(tf.keras.layers.Dense(layer_list[-1], \
                                    activation='tanh',\
                                    kernel_initializer=init))
    
    model.add(tf.keras.layers.Lambda(lambda x: x * 5.))
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