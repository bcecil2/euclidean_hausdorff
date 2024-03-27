import numpy as np
from geometry import make_rot_mx, make_refl_mx


class Transformation(object):

    def __init__(self, delta, rho, is_refl_nontriv, inv_order=False):
        """

        :param delta: translation vector, k-array
        :param rho: rotation vector, (k choose 2)-array
        :param is_refl_nontriv: whether to reflect, boolean
        """
        self.k = len(delta)
        self.delta = np.array(delta)
        self.rho = np.array(rho)
        self.is_refl_nontriv = bool(is_refl_nontriv)
        self.inv_order = inv_order

    def apply(self, coords):
        refl_mx = make_refl_mx(self.k, nontriv=self.is_refl_nontriv)
        rot_mx = make_rot_mx(self.k, self.rho)

        if self.inv_order:
            transformed_coords = (coords + self.delta) @ rot_mx.T @ refl_mx.T
        else:
            transformed_coords = coords @ refl_mx.T @ rot_mx.T + self.delta

        return transformed_coords

    def invert(self):
        return Transformation(-self.delta, -self.rho, self.is_refl_nontriv, inv_order=True)
