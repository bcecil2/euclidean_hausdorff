import numpy as np
from scipy.spatial.transform import Rotation

def make_rot_mx(k, rho):
    """
    Compile rotation matrix for a given angle.

    :param k: dimension
    :param rho: scalar angle (1-array) if k=2 or rotation vector (3-array) if k=3
    :return: (k×k)-array
    """
    if k == 2:
        theta, = rho
        rot_mx = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    else:   # k == 3
        rot_mx = Rotation.from_rotvec(rho).as_matrix()

    return rot_mx


def make_refl_mx(k, nontriv=True):
    """
    Compile reflection matrix for either the trivial or a fixed non-trivial reflection.

    :param k: dimension
    :param nontriv: whether the reflection is non-trivial
    :return: (k×k)-array
    """
    refl_mx = np.eye(k)
    if nontriv:
        refl_mx[-1, -1] = -1
    
    return refl_mx
