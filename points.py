import numpy as np

from transformation import Transformation
from geometry import rot2d, refl2d


class Points(object):

    def __init__(self, points):
        self.n, self.n_dim = points.shape
        self.coords = points

    def reflect(self, is_reflection):
        assert self.n_dim == 2, 'reflection in 3D not implemented'

        return Points(self.coords @ refl2d(is_reflection).T)

    def rotate(self, theta):
        assert self.n_dim == 2, 'rotation in 3D not implemented'

        return Points(self.coords @ rot2d(theta).T)

    def shift(self, delta):
        return Points(self.coords + delta)

    def transform(self, T):
        return self.reflect(T.is_reflection).rotate(T.angle).shift(T.shift)
