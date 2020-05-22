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

    #def __init__(self, pde, network, cond_b, cond_c, cfg):
    def __init__(self, pde, cond_b, cond_c, cfg):

        self.pde = pde
        self.network = cfg['network']
        self.cond_b = cond_b
        self.cond_c = cond_c
        self.cfg = cfg
        if self.cond_b is not None and self.cond_c is not None:
        #if True:
            self.get_collocations()
        
    def train(self):

        criterion = tf.keras.losses.MeanSquaredError()
        optimizer = self.cfg['optimizer']
        train_loss = metrics.Mean()
        
        def compute_loss(y, preds):
            loss = criterion(y, preds)
            return loss

        @tf.function
        def train_step():

            with tf.GradientTape() as tape:
                tape.watch(self.network.trainable_variables)
                
                Xc = tf.concat([self.xc, self.yc], 1)
                Xc = self.preprocess(Xc)
                

                U_net = self.pde(self.network, Xc, self.cfg)
                loss1 = tf.reduce_sum(tf.square(self.uc - U_net))
                #loss1 = tf.reduce_mean(tf.square(self.uc - U_net))

                Xb = tf.concat([self.xb, self.yb], 1)
                Xb = self.preprocess(Xb)
                Ub = self.network(Xb)
                loss2 = tf.reduce_sum(tf.square(self.ub - Ub))
                #loss2 = tf.reduce_mean(tf.square(self.ub - Ub))

                
                Xc_rand = Xc + tf.random.normal(Xc.shape, mean=0.0, stddev=0.002)
                Xc_rand = self.preprocess(Xc_rand)
                Uc = self.network(Xc)
                Uc_rand = self.network(Xc_rand)
                loss3 = tf.reduce_mean(tf.square(Uc - Uc_rand))

                #print('loss1', loss1)
                #print('loss2', loss2)
                #print('loss3', loss3)
                
                #loss_total = loss1 + loss2
                loss_total = loss1 + loss2 + loss3
                #loss_total = loss2

            grads = tape.gradient(loss_total, self.network.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
            train_loss(loss_total)

        for epoch in range(1, self.cfg['epoch'] + 1):
            
            lr = self.cfg['scheduler'](epoch)
            if lr is not None:
                optimizer.lr = lr

            if epoch % 1 == 0:
                self.get_collocations()

            train_step()

            print('Epoch: {}, Cost: {:.7f}'.format(
                epoch+1,
                train_loss.result()
            ))

            if train_loss.result() < 5e-1:
                break

        #https://stackoverflow.com/questions/59029854/use-scipy-optimizer-with-tensorflow-2-0-for-neural-network-training
        #import scipy.optimize as sopt

        #https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize
        #https://www.tensorflow.org/probability/examples/Optimizers_in_TensorFlow_Probability
        #import tensorflow_probability as tfp
        #https://github.com/pierremtb/PINNs-TF2.0/blob/d07587cd63efd08ee826e759e5c5c0bb31edf610/utils/neuralnetwork.py


    def predict(self, X):
        Xc = self.preprocess(X)
        return self.network(Xc)

    def get_collocations(self):

        sp_b, ub = self.cond_b.generate()
        sp_c, uc = self.cond_c.generate()

        #sp_c = sp_c + np.random.randn(sp_c.shape[0], sp_c.shape[1]) / 10.
        #with tf.device("GPU:0"):
        #if True:
        self.ub = tf.convert_to_tensor(ub[:, None])
        self.xb = tf.convert_to_tensor(sp_b[:, 0][:, None])
        self.yb = tf.convert_to_tensor(sp_b[:, 1][:, None])
        self.uc = tf.convert_to_tensor(uc[:, None])
        self.xc = tf.convert_to_tensor(sp_c[:, 0][:, None])
        self.yc = tf.convert_to_tensor(sp_c[:, 1][:, None])

    def preprocess(self, X):
        if False:
            self.bound_l = 0.
            self.bound_u = 1.
            H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 1.0

        if True:
            self.bound_l = -5.
            self.bound_u = 5.
            H = 2.0*(X - self.bound_l)/(self.bound_u - self.bound_l) - 1.0

        return H

    def save(self, path2save):
        self.network.save(path2save)