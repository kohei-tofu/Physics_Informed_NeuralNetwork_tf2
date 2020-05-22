
import numpy as np
import sampler

class base_condition():

    def __init__(self, shapes, N):
        self.shapes = shapes
        self.N = N

    def generate(self):
        return NotImplementedError

class Nothing(base_condition):
    def __init__(self):
        self.N = 0

    def generate(self):
        xb = np.zeros((1, 3), dtype=np.float32)
        ub = np.zeros(1)

        return xb, ub


    
class dirichlet(base_condition):

    def __init__(self, shapes, u_func, N=-1, u_dim = 1):
        super(dirichlet, self).__init__(shapes, N)
        self.u_func = u_func
        self.u_dim = u_dim
        if N == -1:
            self.N = len(self.shapes)

    def generate(self):

        collocation_list = []
        shps = self.shapes.get_list()
        N_ = self.N // len(shps)
        
        if type(self.u_func) is not list:
            for sampler in shps:
                collocation_list.append(sampler.sampling(N_))
            
            if len(shps) == 1:
                collocation = collocation_list[0]
            else:
                collocation = np.concatenate([r for r in collocation_list], 0)
            #print(collocation.shape)
            
            uv_list = self.u_func(collocation)
            
            
        elif len(self.u_func) == len(shps):

            uv_list = []
            for sampler, u in zip(shps, self.u_func):
                
                collocation = sampler.sampling(N_)
                collocation_list.append(collocation)
                uv_list.append(u(collocation))
                
            collocation = np.concatenate([r for r in collocation_list], 0)
            uv_list = np.concatenate([r for r in uv_list], 1)
            
        else:
            print("error")
    
        return collocation, uv_list
