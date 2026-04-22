import numpy as np
from agamoo_ray import Objective


class DF1(Objective):
    def __init__(self, num, obj=1, n_var=2, nt=10, tau=1, taut=20, args=None, verbose=False):
        obj = obj - 1
        n_obj = 2
        self.nt = nt
        self.tau = tau
        self.taut = taut

        bounds = list(zip([0.0]*n_var, [1.0]*n_var))
        super(DF1, self).__init__(num, n_var, n_obj, bounds, obj, args, verbose)

    def evaluate(self, x):
        time = 1 / self.nt * (self.tau // self.taut)
        v = np.sin(0.5 * np.pi * time)
        G = np.abs(v)
        H = 0.75 * v + 1.25
        g = 1 + np.sum((x[:, 1:] - G) ** 2, axis=1)

        if self.obj == 0:
            return x[:, 0]
        elif self.obj == 1:
            return g * (1 - ((x[:, 0]/ g) ** H))
        else:
            raise ValueError('obj')