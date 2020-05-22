
import os
import numpy as np
import tensorflow as tf
import solver
import sampler
import network
import condition
from tensorflow.keras import optimizers
from tensorflow import keras
import util
tf.random.set_seed(22)
np.random.seed(22) 
assert tf.__version__.startswith('2.')
import pylab as pl


def func_burgers2(Re):

    def f(x, y, t):
        temp = (-4 * x + 4 * y - t) * Re / 32.
        temp2 = 4 * (1 + np.exp(temp))
        U = 3. / 4. -  1. / temp2
        V = 3. / 4. +  1. / temp2
        #return [U, V]
        return np.concatenate([U[:, np.newaxis], V[:, np.newaxis]], 1)
    return f



def translate_func_2d(func):
    f = lambda ret : func(ret[:, 0], ret[:, 1], 0) 
    return f



def boundary_term(model, Xc, cfg):
    UV = model(Xc)
    return [UV[:, :(cfg['q']+1)], UV[:, (cfg['q']+1)::]]

#@tf.function
def pde_term(model, Xc, cfg):
    #Xc = tf.concat([self.xc, self.yc], 1)
    #Xc = self.preprocess(Xc)
    x = Xc[:, 0][:, tf.newaxis]
    y = Xc[:, 1][:, tf.newaxis]
    dummy_x1 = tf.ones([Xc.shape[0], cfg['q']], dtype = np.float32)
    dummy_y1 = tf.ones([Xc.shape[0], cfg['q']], dtype = np.float32)
    dummy_x2 = tf.ones([Xc.shape[0], cfg['q']], dtype = np.float32)
    dummy_y2 = tf.ones([Xc.shape[0], cfg['q']], dtype = np.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        tape.watch(dummy_x1)
        tape.watch(dummy_y1)
        tape.watch(dummy_x2)
        tape.watch(dummy_y2)

        X = tf.concat([x, y], 1)
        U_V = model(X)
        U_net = U_V[:, :(cfg['q']+1)]
        V_net = U_V[:, (cfg['q']+1)::]

        U = U_net[:, :-1]
        Gx = tape.gradient(U, x, output_gradients=dummy_x1)
        Ux = tape.gradient(Gx, dummy_x1)
        GUx = tape.gradient(Ux, x, output_gradients=dummy_x1)
        Gy = tape.gradient(U, y, output_gradients=dummy_y1)
        Uy = tape.gradient(Gy, dummy_y1)
        GUy = tape.gradient(Uy, y, output_gradients=dummy_y1)

        V = V_net[:, :-1]
        Hx = tape.gradient(V, x, output_gradients=dummy_x2)
        Vx = tape.gradient(Hx, dummy_x2)
        GVx = tape.gradient(Vx, x, output_gradients=dummy_x2)
        Hy = tape.gradient(V, y, output_gradients=dummy_y2)
        Vy = tape.gradient(Hy, dummy_y2)
        GVy = tape.gradient(Vy, y, output_gradients=dummy_y2)

    Uxx = tape.gradient(GUx, dummy_x1)
    Uyy = tape.gradient(GUy, dummy_y1)
    Vxx = tape.gradient(GVx, dummy_x2)
    Vyy = tape.gradient(GVy, dummy_y2)

    IRK = cfg['IRK']
    dt = cfg['dt']
    Re = cfg['Re']

    F_U = - (U * Ux + V * Uy) + (Uxx + Uyy) / Re
    F_V = - (U * Vx + V * Vy) + (Vxx + Vyy) / Re

    U0_pde = U_net - dt * tf.matmul(F_U, IRK)# (N, q) * (q+1, q)T -> (N, q+1)
    V0_pde = V_net - dt * tf.matmul(F_V, IRK)# (N, q) * (q+1, q)T -> (N, q+1)

    return [U0_pde, V0_pde]
    #pde = tf.concat([U0_pde, V0_pde], 1)
    #return pde


def train(cfg):

    func = cfg['solution']
    func_init = translate_func_2d(func)
    
    #wiki
    upper = 1.0
    upper = 0.5
    line1 = sampler.Line_AxisX_2d(0, 0, upper)
    line2 = sampler.Line_AxisY_2d(0, 0, upper)
    line3 = sampler.Line_AxisX_2d(upper, 0, upper)
    line4 = sampler.Line_AxisY_2d(upper, 0, upper)
    #cond_b = sampler.dirichlet([line1, line2, line3, line4], 0, Nb)
    shapes_b = sampler.object_shapes([line1, line2, line3, line4])
    cond_b = condition.dirichlet(shapes_b, func_init, cfg['Nb'], 1)
    #cond_b = sampler.Nothing()
    space = sampler.Rectangle([0, 0], [1, 1], 1e-6)
    shapes_i = sampler.object_shapes([space])
    cond_0 = condition.dirichlet(shapes_i, func_init, cfg['N0'], 2)

    pinn = solver.PINN(pde_term, boundary_term, cond_b, cond_0, cfg)
    pinn.train()
    pinn.save()


def evaluate(cfg):
    
    func = cfg['solution']
    func_init = translate_func_2d(func)

    pinn = solver.PINN(pde_term, boundary_term, None, None, cfg)

    upper = 0.5
    space_test = sampler.Rectangle([0, 0], [upper, upper], 0.0)

    N_test = 200
    xy = space_test.sampling_diagonal(N_test)
    x_test = xy[:, 0][:, None]
    y_test = xy[:, 1][:, None]
    print('x_test, y_test', x_test.shape, y_test.shape)


    U_true, V_true = util.make_solution_2d_uv([x_test, y_test], cfg['IRK_time'], func)
    U_pred, V_pred = np.copy(U_true), np.copy(V_true)

    U_pred, V_pred = util.make_predict_2d_uv(x_test, y_test, pinn)


    print('U_true.shape', U_true.shape)

    u_abs_max = np.max(np.abs(U_true))
    u_true_max = np.max(U_true) - 0.0
    u_true_min = np.min(U_true)

    v_abs_max = np.max(np.abs(V_true))
    v_true_max = np.max(V_true) - 0.0
    v_true_min = np.min(V_true)


    U_ErrMax = np.max(np.abs(U_true - U_pred))+0.001
    V_ErrMax = np.max(np.abs(V_true - V_pred))+0.001
    print('U_ErrMax, V_ErrMax', U_ErrMax, V_ErrMax)


    pl.figure()
    for loop in range(0, 500, 50):

        print('t', loop, cfg['IRK_time'][loop])
        pl.clf()

        pl.subplot(231)
        pl.imshow(U_true[:, :, loop])
        pl.colorbar()
        pl.clim([u_true_min, u_true_max])

        pl.subplot(232)
        pl.imshow(U_pred[:, :, loop])
        pl.colorbar()
        pl.clim([u_true_min, u_true_max])

        pl.subplot(233)
        pl.imshow(np.abs(U_true[:, :, loop] - U_pred[:, :, loop]))
        pl.colorbar()
        pl.clim([0, U_ErrMax])


        pl.subplot(234)
        pl.imshow(V_true[:, :, loop])
        pl.colorbar()
        pl.clim([v_true_min, v_true_max])

        pl.subplot(235)
        pl.imshow(V_pred[:, :, loop])
        pl.colorbar()
        pl.clim([v_true_min, v_true_max])

        pl.subplot(236)
        pl.imshow(np.abs(V_true[:, :, loop] - V_pred[:, :, loop]))
        pl.colorbar()
        pl.clim([0, V_ErrMax])


        fname = str(loop)
        fname = cfg['path2save'] + '0' * (5 - len(fname)) + fname + '.png'
        pl.savefig(fname)



import scheduler
def get_config1():
    
    cfg = {}
    cfg['Re'] = 20
    cfg['solution'] = func_burgers2(cfg['Re'])

    #cfg['epoch'] = 5000000
    cfg['epoch'] = 300000
    cfg['mode'] = 'fst'
    #cfg['mode'] = 'ctn'

    #scheduler = scheduler.step2
    #sch_function = scheduler.step1
    sch_function = scheduler.const(4e-3)

    #cfg['optimizer'] = optimizers.SGD(learning_rate=scheduler(0), momentum=0.9)
    cfg['optimizer'] = optimizers.Adam(learning_rate=sch_function(0))
    #cfg['optimizer'] = optimizers.Adamax(learning_rate=scheduler(0))
    cfg['scheduler'] = sch_function

    cfg['q'] = 500
    #cfg['layers'] = [2, 50, 50, 50, 50, 50, (cfg['q']+1) * 2]
    cfg['layers'] = [2, 50, 50, 50, 50, 50, 50, (cfg['q']+1) * 2]


    cfg['N0'] = 200
    cfg['Nb'] = 200
    cfg['dt'] = 0.1
    
    cfg['IRK'], cfg['IRK_time'] = util.get_times(cfg['q'], cfg['dt'])
    cfg['IRK'] = tf.constant(cfg['IRK'])
    cfg['path2save'] = './result/burgers/'
    cfg['fname_model'] = 'network.h5'

    path = './result/'
    util.mkdir(path)
    util.mkdir(cfg['path2save'])

    return cfg
    

if __name__ == '__main__':

    #np.random.seed(1234)
    #tf.set_random_seed(1234)

    print('start')

    cfg = get_config1()
    if cfg['mode'] == 'fst':
        #cfg['network'] = network.get_linear(cfg['layers'])
        cfg['network'] = network.get_linear2(cfg['layers'])
    elif cfg['mode'] == 'ctn':
        cfg['network'] = keras.models.load_model(cfg['path2save'] + cfg['fname_model'])
    cfg['network'].summary()


    """

    """

    train(cfg)
    
    evaluate(cfg)

    print('end')