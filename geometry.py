import numpy as np
from scipy.spatial.transform import Rotation


def rot2d(theta,invert=False):
      if invert:
          theta = -theta
      return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def rot3d(n,theta,invert=False):
    #page 5 from here https://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
    n1,n2,n3 = n
    c,s = np.cos(theta),np.sin(theta)
    row1 = [c + n1**2*(1-c), n1*n2*(1-c) - n3*s, n1*n3*(1-c) + n2*s]
    row2 = [n1*n2*(1-c) + n3*s, c + n2**2*(1-c), n2*n3*(1-c)-n1*s]
    row3 = [n1*n3*(1-c)-n2*s, n2*n3*(1-c) + n1*s, c + n3**2*(1-c)]
    R = np.array([row1,row2,row3])
    if invert:
        return np.linalg.inv(R)
    else:
        return R


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
