
import numpy as np

def get_times(q, dt):
    tmp = np.float32(np.loadtxt('./IRK/Butcher_IRK%d.txt' % (q), ndmin = 2))
    weights = np.reshape(tmp[0:q**2+q], (q+1,q))
    #weights = dt * np.reshape(tmp[0:q**2+q], (q+1,q))
    times = dt * tmp[q**2+q:]
    #times = tmp[q**2+q:]
    times = np.array([0.] + times[:, 0].tolist())[:, None]
    return weights.T, times

def make_solution_2d(xy, t, func):
    x = xy[0]
    y = xy[1]
    #dt = t[1, 0] - t[0, 0]
    #dx = x[1, 0] - x[0, 0]
    X, Y, T = np.meshgrid(x, y, t)
    T = T.flatten()[:, None]
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    U = func(X, Y, T)
    print(U.shape)
    umap = np.reshape(U, (x.shape[0], y.shape[0], t.shape[0]))
    #umap = np.reshape(U, (t.shape[0], x.shape[0], y.shape[0]))
    return umap

def make_predict_2d(x, y, model):

    x_star, y_star = np.meshgrid(x, y)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    X = np.concatenate((x_star, y_star), 1)
    umap_pred = model.predict(X)
    umap_pred = np.reshape(umap_pred, (x.shape[0], y.shape[0], model.cfg['q']+1))
    return umap_pred
