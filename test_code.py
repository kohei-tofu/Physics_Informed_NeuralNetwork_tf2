
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
