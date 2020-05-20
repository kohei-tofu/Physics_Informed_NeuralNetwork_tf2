


import numpy as np
import tensorflow as tf
import solver
import sampler
import network
import condition




def make_solution(xy, t, func):
    x = xy[0]
    y = xy[1]
    dt = t[1, 0] - t[0, 0]
    dx = x[1, 0] - x[0, 0]
    X, Y, T = np.meshgrid(x, y, t)
    T = T.flatten()[:, None]
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    U = func(X, Y, T)
    print(U.shape)
    umap = np.reshape(U, (x.shape[0], y.shape[0], t.shape[0]))
    #umap = np.reshape(U, (t.shape[0], x.shape[0], y.shape[0]))
    return umap

def make_predict(x, y, model):

    x_star, y_star = np.meshgrid(x, y)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    X = np.concatenate((x_star, y_star), 1)
    umap_pred = model.predict(X)
    umap_pred = np.reshape(umap_pred, (x.shape[0], y.shape[0], model.cfg['q']+1))
    return umap_pred



def func1(c, D):
    f = lambda x, y, t : np.exp(- 2 * t) * np.sin(x) * np.sin(y)
    return f


def func2(c, D):
    m = 3
    n = 4
    l2 = D * ((m * np.pi)**2 + (n * np.pi)**2)
    f = lambda x, y, t : np.exp(-l2 * t) * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    return f


def func3(c, D):
    def f(x, y, t):
        ret = 0
        m_list = [9, 4]
        n_list = [3, 1]
        A = np.array([300, 500, 700, 200]) * 1e12
        i = 0
        for m in m_list:
            for n in n_list:
                l2 = D * ((m * np.pi)**2 + (n * np.pi)**2)
                ret += A[i] * np.sin(m * np.pi, x) * np.sin(n * np.pi * y) * np.exp(-l2 * t)
                i += 1
        return ret
    return f


def translate_func_2d(func):
    f = lambda ret : func(ret[:, 0], ret[:, 1], 0)
    return f




def get_times(q, dt):
    tmp = np.float32(np.loadtxt('./IRK/Butcher_IRK%d.txt' % (q), ndmin = 2))
    weights = np.reshape(tmp[0:q**2+q], (q+1,q))
    times = dt * tmp[q**2+q:]
    times = np.array([0.] + times[:, 0].tolist())[:, None]
    return weights.T, times


#@tf.function
def net_U1(model, Xc, cfg):
    #Xc = tf.concat([self.xc, self.yc], 1)
    #Xc = self.preprocess(Xc)
    x = Xc[:, 0][:, tf.newaxis]
    y = Xc[:, 1][:, tf.newaxis]
    with tf.GradientTape() as g1, tf.GradientTape() as g2:
        with tf.GradientTape() as g3, tf.GradientTape() as g4:
            g1.watch(x)
            g2.watch(x)
            g3.watch(y)
            g4.watch(y)
            X = tf.concat([x, y], 1)
            U_net = model(X)
            U = U_net[:, :-1]

    U_y = g3.batch_jacobian(U, y)[:, :, 0]
    U_yy = g4.batch_jacobian(U_y, y)[:, :, 0]
    U_x = g1.batch_jacobian(U, x)[:, :, 0]
    U_xx = g2.batch_jacobian(U_x, x)[:, :, 0]
            
    #C = cfg['C']
    D = cfg['D']
    IRK = cfg['IRK']
    dt = cfg['dt']
    
    
    #F = - c * (U_x + U_y) + D * (U_xx + U_yy)
    F = D * (U_xx + U_yy)
    #U0_pde = U_net - dt * tf.matmul(F, IRK.T)# (N, q) * (q+1, q)T -> (N, q+1)
    U0_pde = U_net - dt * tf.matmul(F, IRK)# (N, q) * (q+1, q)T -> (N, q+1)
    return U0_pde


@tf.function
def net_U0(model, Xc, cfg):
    #Xc = tf.concat([self.xc, self.yc], 1)
    #Xc = self.preprocess(Xc)
    x = Xc[:, 0][:, tf.newaxis]
    y = Xc[:, 1][:, tf.newaxis]

    with tf.GradientTape() as g3, tf.GradientTape() as g4:
        with tf.GradientTape() as g1, tf.GradientTape() as g2:
            g1.watch(x)
            g2.watch(y)
            g3.watch(x)
            g4.watch(y)
            X = tf.concat([x, y], 1)
            U_net = model(X)
            U = U_net[:, :-1]
        U_x = g1.batch_jacobian(U, x)[:, :, 0]
        U_y = g2.batch_jacobian(U, y)[:, :, 0]
    U_xx = g3.batch_jacobian(U_x, x)[:, :, 0]
    U_yy = g4.batch_jacobian(U_y, y)[:, :, 0]        
    #print(U_y, U_y.shape, 'U_y')
    #print(U_x, U_x.shape, 'U_x')
    #print(U_yy, U_yy.shape, 'U_yy')
    #print(U_xx, U_xx.shape, 'U_xx')
    #C = cfg['C']
    D = cfg['D']
    IRK = cfg['IRK']
    dt = cfg['dt']
    
    
    #F = - c * (U_x + U_y) + D * (U_xx + U_yy)
    F = D * (U_xx + U_yy)
    #U0_pde = U_net - dt * tf.matmul(F, IRK.T)# (N, q) * (q+1, q)T -> (N, q+1)
    U0_pde = U_net - dt * tf.matmul(F, IRK)# (N, q) * (q+1, q)T -> (N, q+1)
    return U0_pde


def main1():

    cfg = {}
    cfg['epoch'] = 50000
    N0 = 250
    Nb = 50
    #cfg['dt'] = dt = 1e-3
    cfg['dt'] = dt = 1e-3
    cfg['q'] = q = 500

    cfg['c'] = c = 0.0
    cfg['D'] = D = 8.0
    cfg['IRK'], cfg['IRK_time'] = get_times(q, dt)

    print('IRK', cfg['IRK'])
    func = func2(c, D)
    func_init = translate_func_2d(func)

    

    #wiki
    upper = 1.0
    line1 = sampler.Line_AxisX_2d(0, 0, upper)
    line2 = sampler.Line_AxisY_2d(0, 0, upper)
    line3 = sampler.Line_AxisX_2d(upper, 0, upper)
    line4 = sampler.Line_AxisY_2d(upper, 0, upper)
    #cond_b = sampler.dirichlet([line1, line2, line3, line4], 0, Nb)
    shapes_b = sampler.object_shapes([line1, line2, line3, line4])
    cond_b = condition.dirichlet(shapes_b, func_init, Nb, 1)
    #cond_b = sampler.Nothing()


    space = sampler.Rectangle([0, 0], [1, 1], 1e-6)
    space_test = sampler.Rectangle([0, 0], [1, 1], 0.0)
    shapes_i = sampler.object_shapes([space])
    cond_0 = condition.dirichlet(shapes_i, func_init, N0, 1)


    #layers = [2, 50, 50, 50, 50, 50, 50, q+1]
    layers = [2, 50, 50, 50, q+1]
    net = network.get_linear(layers)
    #mdl = pde.PINN_Diffusion
    #main1 = main1_2d

    pinn = solver.PINN(net_U0, net, cond_b, cond_0, cfg)
    #print(pinn.network.weights)
    pinn.train()
    #print(pinn.network.weights)

    N_test = 200
    xy = space_test.sampling_diagonal(N_test)
    x_test = xy[:, 0][:, None]
    y_test = xy[:, 1][:, None]

    y_true = make_solution([x_test, y_test], cfg['IRK_time'], func)
    y_pred = np.copy(y_true)
    y_pred = make_predict(x_test, y_test, pinn)
    print(y_true.shape, y_pred.shape)

    y_abs_max = np.max(np.abs(y_true))
    #y_true_max = np.max(y_true) - 0.2
    y_true_max = np.max(y_true) - 0.0
    y_true_min = np.min(y_true)

    y_ErrMax = np.max(np.abs(y_true - y_pred))+0.001
    print('y_ErrMax', y_ErrMax)

    import pylab as pl
    pl.figure()
    for loop in range(0, 500, 50):

        #print('t', t[loop])
        #y_true = main2([x_test, y_test], t[loop], func)
        #y_pred = np.copy(y_true)
        #print(np.sum(np.abs(y_pred[loop] - y_pred[loop+50])))

        pl.clf()

        pl.subplot(131)
        pl.imshow(y_true[:, :, loop])
        pl.colorbar()
        pl.clim([y_true_min, y_true_max])


        pl.subplot(132)
        pl.imshow(y_pred[:, :, loop])
        pl.colorbar()
        pl.clim([y_true_min, y_true_max])

        pl.subplot(133)
        pl.imshow(np.abs(y_true[:, :, loop] - y_pred[:, :, loop]))
        pl.colorbar()
        pl.clim([0, y_ErrMax])
        #pl.show()

        fname = str(loop)
        fname = './result/' + '0' * (5 - len(fname)) + fname + '.png'
        pl.savefig(fname)

        
if __name__ == '__main__':

    #np.random.seed(1234)
    #tf.set_random_seed(1234)

    print('start')

    main1()

    print('end')