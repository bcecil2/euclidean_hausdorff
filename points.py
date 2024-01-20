import numpy as np

from geometry import rot2d, refl2d


class Points(object):

    def __init__(self, points):
        self.n, self.n_dim = points.shape
        self.coords = points

    def reflect(self, nontriv):
        assert self.n_dim == 2, 'reflection in 3D not implemented'

        return Points(self.coords @ refl2d(nontriv).T)

    def rotate(self, theta):
        assert self.n_dim == 2, 'rotation in 3D not implemented'

        return Points(self.coords @ rot2d(theta).T)

    def shift(self, delta):
        return Points(self.coords + delta)

    def transform(self, nontriv_refl=None, theta=None, delta=None):
        if nontriv_refl is None:
            nontriv_refl = False
        if theta is None:
            theta = 0
        if delta is None:
            delta = np.zeros(self.n_dim)

        return self.reflect(nontriv_refl).rotate(theta).shift(delta)
