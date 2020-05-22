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

            print('Epoch: {}, Cost: {:.7f}'.format(
                epoch+1,
                train_loss.result()
            ))

            if train_loss.result() < 5e-1:
                break

            if epoch % 50000 == 0:
                self.save(self.cfg['path2save'])
                print('saved')


    def train_order2(self):
        import tensorflow_probability as tfp
        #https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
        func = self.function_factory()

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.network.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=500)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)


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

        
    #def function_factory(model, loss, train_x, train_y):
    def function_factory(self):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.
        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(self.network.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                self.network.trainable_variables[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
            params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            self.get_collocations()
            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                #loss_value = loss(model(train_x, training=True), train_y)

                #tape.watch(model.trainable_variables)
                Xc = tf.concat([self.xc, self.yc], 1)
                Xc = self.preprocess(Xc)
                
                U_net = self.pde(self.network, Xc, self.cfg)
                loss1 = tf.reduce_sum(tf.square(self.uc - U_net))
                Xb = tf.concat([self.xb, self.yb], 1)
                Xb = self.preprocess(Xb)
                Ub = self.network(Xb)
                loss2 = tf.reduce_sum(tf.square(self.ub - Ub))
                loss_total = loss1 + loss2
            

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_total, self.network.trainable_variables)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            f.iter.assign_add(1)
            tf.print("Iter:", f.iter, "loss:", loss_total)

            return loss_total, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters

        return f
