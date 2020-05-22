
#@tf.function
def net_U0(model, Xc, cfg):
    x = Xc[:, 0][:, tf.newaxis]
    y = Xc[:, 1][:, tf.newaxis]

    with tf.GradientTape() as g3, tf.GradientTape() as g4:
        g3.watch(x)
        g4.watch(y)
        with tf.GradientTape() as g1, tf.GradientTape() as g2:
            g1.watch(x)
            g2.watch(y)
            X = tf.concat([x, y], 1)
            U_net = model(X)
            U = U_net[:, :-1]
        U_x = g1.batch_jacobian(U, x)[:, :, 0]
        U_y = g2.batch_jacobian(U, y)[:, :, 0]
        print(U_x, U_x.shape, 'U_x')
        print(U_y, U_y.shape, 'U_y')
    U_xx = g3.batch_jacobian(U_x, x)[:, :, 0]
    U_yy = g4.batch_jacobian(U_y, y)[:, :, 0]
    print(U_xx, U_xx.shape, 'U_xx')
    print(U_yy, U_yy.shape, 'U_yy')
    
    D = cfg['D']
    IRK = cfg['IRK']
    dt = cfg['dt']
    #F = - c * (U_x + U_y) + D * (U_xx + U_yy)
    F = D * (U_xx + U_yy)
    #U0_pde = U_net - dt * tf.matmul(F, IRK.T)# (N, q) * (q+1, q)T -> (N, q+1)
    U0_pde = U_net - dt * tf.matmul(F, IRK)# (N, q) * (q+1, q)T -> (N, q+1)
    return U0_pde


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
