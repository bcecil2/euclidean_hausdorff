import numpy as np


class Transformation(object):

    def __init__(self, angle, shift, is_reflection):
        """

        :param angle: (n_dim - 1)-array
        :param shift:  (n_dim)-array
        :param is_reflection: boolean
        """
        self.angle = np.array(angle)
        self.shift = np.array(shift)
        self.is_reflection = bool(is_reflection)

    def invert(self):
        return Transformation(-self.angle, -self.shift, not self.is_reflection)
