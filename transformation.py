import numpy as np
from geometry import rot3d,rot2d

class Transformation(object):

    def __init__(self, delta, theta, reflection, invert=False):
        """

        :param delta: translation amount, k-array
        :param theta: rotation amount, (k-1)-array
        :param reflection: whether to reflect, boolean
        """
        self.delta = np.array(delta)
        self.k = len(theta)
        if self.k == 1:
            self.rot = rot2d(np.array(theta), invert)
        else:
            self.rot = rot3d(theta[0], theta[1], invert)
        self.theta = theta
        self.reflection = bool(reflection)


    def invert(self):
        return Transformation(-self.delta, self.theta, self.reflection, invert=True)
