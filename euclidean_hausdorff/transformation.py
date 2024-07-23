import numpy as np
from scipy.spatial.transform import Rotation

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
        refl_mx = self.make_refl_mx(nontriv=self.is_refl_nontriv)
        rot_mx = self.make_rot_mx()

        if self.inv_order:
            transformed_coords = (coords + self.delta) @ rot_mx.T @ refl_mx.T
        else:
            transformed_coords = coords @ refl_mx.T @ rot_mx.T + self.delta

        return transformed_coords

    def invert(self):
        return Transformation(-self.delta, -self.rho, self.is_refl_nontriv, inv_order=True)


    def make_refl_mx(self,nontriv=True):
        """
        Compile reflection matrix for either the trivial or a fixed non-trivial reflection.

        :param k: dimension
        :param nontriv: whether the reflection is non-trivial
        :return: (k×k)-array
        """
        refl_mx = np.eye(self.k)
        if nontriv:
            refl_mx[-1, -1] = -1

        return refl_mx

    def make_rot_mx(self):
        """
        Compile rotation matrix for a given angle.

        :param k: dimension
        :param rho: scalar angle (1-array) if k=2 or rotation vector (3-array) if k=3
        :return: (k×k)-array
        """
        if self.k == 2:
            theta, = self.rho
            rot_mx = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
        else:  # k == 3
            rot_mx = Rotation.from_rotvec(self.rho).as_matrix()

        return rot_mx


