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
import pylab as pl
import shutil
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.random.set_seed(22)
np.random.seed(22) 
assert tf.__version__.startswith('2.')



def func0(c, D):
    f = lambda x, y, t : (np.exp(- 2 * t) * np.sin(x) * np.sin(y))[:, np.newaxis]
    return f

def func1(c, D):
    def f(x, y, t):
        
        l2 = D * np.square(np.pi) * (1 + 1)
        u = np.exp(-l2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
        return u[:, np.newaxis]
    return f

def func2(c, D):
    def f(x, y, t):
        m = 3
        n = 4
        #l2 = (D**2) * ((m * np.pi)**2 + (n * np.pi)**2)
        #l2 = D * ((m * np.pi / 1.)**2 + (n * np.pi / 1.)**2)
        l2 = D * np.square(np.pi) * ((m / 1.)**2 + (n / 1.)**2)
        u = np.exp(-l2 * t) * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
        return u[:, np.newaxis]

    #f = lambda x, y, t : np.exp(-l2 * t) * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
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
        return ret[:, np.newaxis]
    return f


def translate_func_2d(func):
    f = lambda ret : func(ret[:, 0], ret[:, 1], 0)
    return f
    

def boundary_term(model, Xc, cfg):
    U = model(Xc)
    return [U]

#@tf.function
def pde_term(model, Xc, cfg):
    #Xc = tf.concat([self.xc, self.yc], 1)
    #Xc = self.preprocess(Xc)
    x = Xc[:, 0][:, tf.newaxis]
    y = Xc[:, 1][:, tf.newaxis]
    dummy_x = tf.ones([Xc.shape[0], cfg['q']], dtype = np.float32)
    dummy_y = tf.ones([Xc.shape[0], cfg['q']], dtype = np.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        tape.watch(dummy_x)
        tape.watch(dummy_y)

        X = tf.concat([x, y], 1)
        U_net = model(X)
        U = U_net[:, :-1]

        Gx = tape.gradient(U, x, output_gradients=dummy_x)
        Ux = tape.gradient(Gx, dummy_x)
        GUx = tape.gradient(Ux, x, output_gradients=dummy_x)
        Gy = tape.gradient(U, y, output_gradients=dummy_y)
        Uy = tape.gradient(Gy, dummy_y)
        GUy = tape.gradient(Uy, y, output_gradients=dummy_y)

    Uxx = tape.gradient(GUx, dummy_x)
    Uyy = tape.gradient(GUy, dummy_y)
    D = cfg['D']
    IRK = cfg['IRK']
    dt = cfg['dt']
    F = D * (Uxx + Uyy)
    U0_pde = U_net - dt * tf.matmul(F, IRK)# (N, q) * (q+1, q)T -> (N, q+1)
    return [U0_pde]


def train(cfg):

    func = cfg['solution']
    func_init = translate_func_2d(func)
    
    #wiki
    upper = 1.0
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
    cond_0 = condition.dirichlet(shapes_i, func_init, cfg['N0'], 1)

    #pinn = solver.PINN(net_U0, cond_b, cond_0, cfg)
    pinn = solver.PINN(pde_term, boundary_term, cond_b, cond_0, cfg)
    pinn.train()
    #pinn.train_order2()

    pinn.save()


def evaluate(cfg):
    
    func = cfg['solution']
    func_init = translate_func_2d(func)

    
    pinn = solver.PINN(pde_term, boundary_term, None, None, cfg)

    N_test = 200
    space_test = sampler.Rectangle([0, 0], [1, 1], 0.0)
    xy = space_test.sampling_diagonal(N_test)
    x_test = xy[:, 0][:, None]
    y_test = xy[:, 1][:, None]

    y_true = util.make_solution_2d([x_test, y_test], cfg['IRK_time'], func)
    y_pred = np.copy(y_true)
    y_pred = util.make_predict_2d(x_test, y_test, pinn)
    print(y_true.shape, y_pred.shape)

    y_abs_max = np.max(np.abs(y_true))
    #y_true_max = np.max(y_true) - 0.2
    y_true_max = np.max(y_true) - 0.0
    y_true_min = np.min(y_true)

    y_ErrMax = np.max(np.abs(y_true - y_pred))+0.001
    print('y_ErrMax', y_ErrMax)


    util.delete_files(cfg['path2save'] + '*.png')
    
    pl.figure()
    for loop in range(0, cfg['q'], cfg['q']//5):

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
        fname = cfg['path2save'] + '0' * (5 - len(fname)) + fname + '.png'
        pl.savefig(fname)



import scheduler
def get_config1(args):
    
    cfg = {}
    cfg['c'] = c = 0.0
    cfg['D'] = D = 8.0
    cfg['solution'] = func1(cfg['c'], cfg['D'])

    cfg['epoch'] = args.epoch

    #scheduler = scheduler.step2
    sch_function = scheduler.step1
    sch_function = scheduler.const(1e-4)
    #sch_function = scheduler.const(1e-5)
    #sch_function = scheduler.const(1e-2)
    #cfg['optimizer'] = optimizers.SGD(learning_rate=scheduler(0), momentum=0.9)
    cfg['optimizer'] = optimizers.Adam(learning_rate=sch_function(0))
    #cfg['optimizer'] = optimizers.Adamax(learning_rate=scheduler(0))
    cfg['scheduler'] = sch_function

    #cfg['q'] = 100
    cfg['q'] = 500
    #cfg['layers'] = [2, 50, 50, cfg['q']+1]
    #cfg['layers'] = [2, 50, 50, 50, cfg['q']+1]
    cfg['layers'] = [2, 50, 50, 50, 50, 50, cfg['q']+1]
    #cfg['mode'] = 'fst'
    cfg['mode'] = args.mode

    cfg['N0'] = 25000
    cfg['Nb'] = 20000
    cfg['N0'] = 200
    cfg['Nb'] = 200
    #cfg['dt'] = 1e-3
    cfg['dt'] = 1e-4
    #cfg['q'] = 500
    #cfg['q'] = 20
    
    
    cfg['IRK'], cfg['IRK_time'] = util.get_times(cfg['q'], cfg['dt'])
    cfg['IRK'] = tf.constant(cfg['IRK'])
    cfg['path2save'] = './result/diffusion_1/'
    cfg['fname_model'] = 'network.h5'

    path = './result/'
    util.mkdir(path)
    util.mkdir(cfg['path2save'])

    
    return cfg
    


def get_config2(args):
    
    cfg = {}
    cfg['c'] = c = 0.0
    cfg['D'] = D = 8.0
    cfg['solution'] = func2(cfg['c'], cfg['D'])


    cfg['epoch'] = args.epoch

    #scheduler = scheduler.step2
    sch_function = scheduler.step([5000, 14000], [1e-3, 1e-4, 1e-5])
    #sch_function = scheduler.const(1e-4)
    #sch_function = scheduler.const(1e-5)
    #sch_function = scheduler.const(1e-2)
    #cfg['optimizer'] = optimizers.SGD(learning_rate=scheduler(0), momentum=0.9)
    cfg['optimizer'] = optimizers.Adam(learning_rate=sch_function(0))
    #cfg['optimizer'] = optimizers.Adamax(learning_rate=scheduler(0))
    cfg['scheduler'] = sch_function

    #cfg['q'] = 100
    cfg['q'] = 500
    #cfg['layers'] = [2, 50, 50, cfg['q']+1]
    #cfg['layers'] = [2, 50, 50, 50, cfg['q']+1]
    cfg['layers'] = [2, 50, 50, 50, 50, 50, cfg['q']+1]
    cfg['mode'] = args.mode
    #cfg['mode'] = 'ctn'

    cfg['N0'] = 200
    cfg['Nb'] = 200
    #cfg['dt'] = 1e-3
    #cfg['dt'] = 1e-4
    cfg['dt'] = 1e-4
    #cfg['q'] = 500
    #cfg['q'] = 20
    
    
    cfg['IRK'], cfg['IRK_time'] = util.get_times(cfg['q'], cfg['dt'])
    cfg['IRK'] = tf.constant(cfg['IRK'])
    cfg['path2save'] = './result/diffusion_2/'
    cfg['fname_model'] = 'network.h5'

    path = './result/'
    util.mkdir(path)
    util.mkdir(cfg['path2save'])

    
    return cfg

if __name__ == '__main__':
    import argparse
    #np.random.seed(1234)
    #tf.set_random_seed(1234)

    print('start')
    

    parser = argparse.ArgumentParser(description='solve 2d diffusion using neural netwrok')
    parser.add_argument('--epoch', '-EPOCH', type=int, default=60000, help='epoch')
    parser.add_argument('--job', '-J', type=str, default='evaluate', help='what job are you going to do? train or evaluate')
    parser.add_argument('--mode', '-M', type=str, default='ctn', help='train from previous result')
    args = parser.parse_args()

    #
    #cfg = get_config1(args)
    cfg = get_config2(args)
   
    #
    print(args.job)
    if args.job == 'train':
        print('train')
        if cfg['mode'] == 'fst':
            cfg['network'] = network.get_linear1(cfg['layers'])
            #cfg['network'] = network.get_linear2(cfg['layers'])
        elif cfg['mode'] == 'ctn':
            cfg['network'] = keras.models.load_model(cfg['path2save'] + cfg['fname_model'])
            pass
        cfg['network'].summary()

        train(cfg)
    #
    #elif args.mode == 'evaluate':
    print('evaluate')
    cfg['network'] = keras.models.load_model(cfg['path2save'] + cfg['fname_model'])
    cfg['network'].summary()

    evaluate(cfg)

    print('end')