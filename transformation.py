import numpy as np


class Transformation(object):

    def __init__(self, delta, theta, reflection):
        """

        :param delta: translation amount, k-array
        :param theta: rotation amount, (k-1)-array
        :param reflection: whether to reflect, boolean
        """
        self.delta = np.array(delta)
        self.theta = np.array(theta)
        self.reflection = bool(reflection)

    def invert(self):
        return Transformation(-self.delta, -self.theta, self.reflection)
