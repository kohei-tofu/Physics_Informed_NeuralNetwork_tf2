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
    def __init__(self, pde, net_boundary, cond_b, cond_c, cfg):

        self.pde = pde
        self.net_boundary = net_boundary
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
                
                Xc = self.preprocess(self.Xc_raw)
                Xb = self.preprocess(self.Xb_raw)
                Xc = tf.convert_to_tensor(Xc)
                Xb = tf.convert_to_tensor(Xb)


                pde_term = self.pde(self.network, Xc, self.cfg)
                bc_term = self.net_boundary(self.network, Xb, self.cfg)

                loss1 = 0
                loss2 = 0
                for i, (p, b) in enumerate(zip(pde_term, bc_term)):
                    loss1 += tf.reduce_sum(tf.square(self.uc[:, i][:, tf.newaxis] - p))
                    loss2 += tf.reduce_sum(tf.square(self.ub[:, i][:, tf.newaxis] - b))
                    #loss1 = tf.reduce_sum(tf.square(self.uc - U_net))
                    #loss1 = tf.reduce_mean(tf.square(self.uc - U_net))

                    
                    #loss2 = tf.reduce_sum(tf.square(self.ub[:, i] - Ub))
                    #loss2 = tf.reduce_mean(tf.square(self.ub - Ub))

                
                #Xc_rand = Xc + tf.random.normal(Xc.shape, mean=0.0, stddev=0.002)
                #Xc_rand = self.preprocess(Xc_rand)
                #Uc = self.network(Xc)
                #Uc_rand = self.network(Xc_rand)
                #loss3 = tf.reduce_mean(tf.square(Uc - Uc_rand))

                #print(loss1, loss2, loss3)
                
                loss_total = loss1 + loss2
                #loss_total = loss1 + loss2 + loss3

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

            if epoch % 200 == 0:
                print('Epoch: {}, Cost: {:.7f}'.format(
                    epoch,
                    train_loss.result()
                ))

            if train_loss.result() < 5e-1:
                break

            if epoch % 20000 == 0:
                self.save()
                print('saved')


    def predict(self, X):
        Xc = self.preprocess(X)
        return self.network(Xc)

    def get_collocations(self):

        sp_b, ub = self.cond_b.generate()
        sp_c, uc = self.cond_c.generate()
        self.ub = tf.convert_to_tensor(ub)
        self.uc = tf.convert_to_tensor(uc)
        self.Xb_raw = sp_b
        self.Xc_raw = sp_c

        
        #sp_c = sp_c + np.random.randn(sp_c.shape[0], sp_c.shape[1]) / 10.
        #with tf.device("GPU:0"):
        #if True:
        #self.ub = tf.convert_to_tensor(ub)
        #self.xb = tf.convert_to_tensor(sp_b[:, 0][:, None])
        #self.yb = tf.convert_to_tensor(sp_b[:, 1][:, None])
        #self.uc = tf.convert_to_tensor(uc)
        #self.xc = tf.convert_to_tensor(sp_c[:, 0][:, None])
        #self.yc = tf.convert_to_tensor(sp_c[:, 1][:, None])

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

    def save(self):
        path2save = self.cfg['path2save'] + self.cfg['fname_model']
        self.network.save(path2save)

        