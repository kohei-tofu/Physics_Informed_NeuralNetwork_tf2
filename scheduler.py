


def const(lr):
    def func(epoch): 
        return lr
    return func

def step0(epoch):
    if epoch < 10000:
        return 1e-1
    elif epoch < 20000:
        return 1e-2
    elif epoch < 50000:
        return 1e-3
    else:
        return 1e-4

def step1(epoch):
    if epoch < 1000:
        return 1e-2
    elif epoch < 5000:
        return 1e-3
    elif epoch < 180000:
        return 1e-4
    else:
        return 1e-5

def step2(epoch):
    if epoch < 4000:
        return 2e-2
    elif epoch < 40000:
        return 1e-2
    elif epoch < 80000:
        return 2e-3
    else:
        return 1e-3