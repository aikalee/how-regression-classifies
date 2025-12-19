import numpy as np

class SGDNesterov:
    def __init__(self, params, lr, momentum):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i, (p, g) in enumerate(zip(self.params, grads)):
            v_prev = self.v[i]
            self.v[i] = self.momentum * self.v[i] + g
            p -= self.lr * (self.momentum * v_prev + g)
