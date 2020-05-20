import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

from easydict import EasyDict as edict

class Config:
    #
    #  DATASET
    #
    VAL = edict()
    VAL.EPOCH = 100

    #def preprocess(self, X):
    #    H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 1.0
    #    #H = 2.0 * (X - bound_l) / (bound_u - bound_l) - 1.0
    #    return H
        
class PINN:

    def __init__(self, pde, network, cond_b, cond_c, cfg):

        self.pde = pde
        self.network = network
        self.cond_b = cond_b
        self.cond_c = cond_c
        self.cfg = cfg

    
    def train(self):

        criterion = tf.keras.losses.MeanSquaredError()
        optimizer = optimizers.Adam(learning_rate=1e-2)
        train_loss = metrics.Mean()
        
        def compute_loss(y, preds):
            loss = criterion(y, preds)
            return loss

        #
        #@tf.autograph.experimental.do_not_convert
        @tf.function
        def train_step():

            with tf.GradientTape() as tape:
                tape.watch(self.network.trainable_variables)
                Xc = tf.concat([self.xc, self.yc], 1)
                Xc = self.preprocess(Xc)
                U_net = self.pde(self.network, Xc, self.cfg)
                #U_net = self.net_U0()
                uc_ = np.broadcast_to(self.uc, U_net.shape)
                #loss1 = tf.reduce_sum(tf.square(self.uc - U_net))
                loss1 = tf.reduce_sum(tf.square(uc_ - U_net))

                Xb = tf.concat([self.xb, self.yb], 1)
                Xb = self.preprocess(Xb)
                Ub = self.network(Xb)
                #loss2 = criterion(self.ub, Ub)
                loss2 = tf.reduce_sum(tf.square(self.ub - Ub))

                #print('loss1', loss1)
                #print('loss2', loss2)

                loss_total = loss1 + loss2
                #loss_total = loss2

            grads = tape.gradient(loss_total, self.network.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
            train_loss(loss_total)

        for epoch in range(1, self.cfg['epoch'] + 1):

            self.get_collocations()

            train_step()

            print('Epoch: {}, Cost: {:.7f}'.format(
                epoch+1,
                train_loss.result()
            ))


    def predict(self, X):
        Xc = self.preprocess(X)
        return self.network(Xc)

    def get_collocations(self):

        sp_b, ub = self.cond_b.generate()
        self.ub = ub[:, None]
        self.xb = tf.convert_to_tensor(sp_b[:, 0][:, None])
        self.yb = tf.convert_to_tensor(sp_b[:, 1][:, None])
        
        sp_c, uc = self.cond_c.generate()
        self.uc = uc[:, None]
        self.xc = tf.convert_to_tensor(sp_c[:, 0][:, None])
        self.yc = tf.convert_to_tensor(sp_c[:, 1][:, None])

    def preprocess(self, X):
        self.bound_l = 0.
        self.bound_u = 1.
        #H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 1.0
        H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 2.0
        return H