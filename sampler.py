 
import numpy as np
import types




class object_shapes():

    def __init__(self, shapes_list):
        self.shapes_list = shapes_list

    def get_list(self):
        return self.shapes_list
        
    def generate(self):
        return NotImplementedError
    
    def get_lower(self):
        return 0.
    
    def get_upper(self):
        return 1.


class Points_1d():

    def __init__(self, x):
        self.x = x

    def sampling(self, N):
        ret_x = self.x * np.ones(N)[:, None]
        return ret_x
        
        
class Line_1d():   
    def __init__(self, lower, upper, dx, order_type='rand'):
        #print(lower, upper, dx)
        self.lower = lower + dx
        self.upper = upper - dx

    def sampling(self, N, order_type='rand'):
        if order_type == 'rand':
            temp = (self.upper - self.lower) * np.random.random(N) + self.lower
        elif order_type == 'order':
            temp = np.linspace(self.lower, self.upper, N)
            
        #print('line', temp.shape)
        return temp[:, None]



class Points_2d():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def sampling(self, N):
        ret_x = self.x * np.ones(N)
        ret_y = self.y * np.ones(N)
        return np.concatenate((ret_x[:, None], ret_y[:, None]), 1)
        
        
        
class Rectangle():   
    def __init__(self, lower, upper, dx, order_type='rand'):
        self.lower = lower
        self.upper = upper

        if type(dx) is float:
            self.dx = [dx] * len(lower)
        elif type(dx) is list:
            self.dx = dx

    def sampling(self, N, order_type='rand'):

        if order_type == 'rand':
            ret = []
            for l, u, _dx in zip(self.lower, self.upper, self.dx):
                ret.append(Line_1d(l, u, _dx).sampling(N)[:, 0].tolist())
            ret = np.array(ret).T
        return ret.astype(np.float32)

    def sampling_diagonal(self, N):
        #Diagonal sampling
        ret = []
        for l, u, _dx in zip(self.lower, self.upper, self.dx):
            #temp = np.arange(start = l, stop = u + x_step, step=x_step)
            ret.append(np.linspace(start = l, stop = u, num=N).tolist())
        ret = np.array(ret).T
        return ret.astype(np.float32)

class Line_AxisX_2d():
    #
    # cross to the axis X
    #
    def __init__(self, X, lower_y, upper_y, dx=0, order_type='rand'):
        self.x = X
        
        # range y
        self.lower = lower_y + dx
        self.upper = upper_y - dx

    def sampling(self, N, order_type='rand'):
        ret_x = self.x * np.ones(N)
        if order_type == 'rand':
            ret_y = (self.upper - self.lower) * np.random.random(N) + self.lower
        return np.concatenate((ret_x[:, None], ret_y[:, None]), 1).astype(np.float32)

class Line_AxisY_2d(Line_AxisX_2d):   
    def __init__(self, X, lower, upper, dx=0, order_type='rand'):
        super(Line_AxisY_2d, self).__init__(X, lower, upper, dx, order_type)

    def sampling(self, N, order_type='rand'):
        ret_y = self.x * np.ones(N)
        if order_type == 'rand':
            ret_x = (self.upper - self.lower) * np.random.random(N) + self.lower
        return np.concatenate((ret_x[:, None], ret_y[:, None]), 1).astype(np.float32)




"""

class volume():

    def __init__(self, func_list, init_func, N):
        #super(volume , self).__init__(func_list, N)
        self.init_func = init_func
        self.lower = func_list[0].lower
        self.upper = func_list[0].upper

    def generate(self):

        N_ = self.N // len(self.func_list)

        ret = []
        for f_ in self.func_list:

            ret.append(f_.sampling(N_).tolist())

        ret = np.array(ret)
        ret = np.swapaxes(ret, 1, 2)
        ret = ret.reshape((ret.shape[0] * ret.shape[1], ret.shape[2]))

        return ret, self.init_func(ret)[:, None]

    def generate_diagonal(self):

        N_ = self.N // len(self.func_list)

        ret = []
        for f_ in self.func_list:

            ret.append(f_.sampling_diagonal(N_).tolist())

        ret = np.array(ret)
        ret = np.swapaxes(ret, 1, 2)
        ret = ret.reshape((ret.shape[0] * ret.shape[1], ret.shape[2]))

        return ret, self.init_func(ret)[:, None]

"""

if __name__ == '__main__':

    """

    N = 200
    f_list1 = Line_OnX_2d(5, 0.1, 0.2, 0)
    f_list2 = Line_OnX_2d(5, 0.1, 0.2, 0)
    f_list3 = Line_OnY_2d(7, 0.1, 0.3, 0)
    #print(f_list3.sampling(N))

    cond = dirichlet([f_list1, f_list2, f_list3], 0, N)
    ret, u = cond.generate()

    #print(ret[:, 0])
    #print(ret[:, 1])
    print(ret)
    print(ret.shape, u.shape)


    #if False:
    if True:
        i_func = lambda d : np.sin(np.pi * d[:, 0]) * np.sin(np.pi * d[:, 1])
        #rec = Line(0, 1, 1e-5) + Line(0, 2, 1e-5)
        rec = Rectangle([0, 0], [1, 1], [1e-5, 1e-5])
        i_cond = volume([rec], i_func, 200)
        ret, u = i_cond.generate()
        #ret, u = i_cond.generate_diagonal()


        #print(ret[:, 0])
        #print(ret[:, 1])
        print(ret)
        print(ret.shape, np.max(ret), np.min(ret))
        print(u.shape, np.max(u), np.min(u))

        import pylab as pl
        pl.figure()
        pl.scatter(ret[:, 0], ret[:, 1])
        pl.grid()
        pl.show()
    """
    pass