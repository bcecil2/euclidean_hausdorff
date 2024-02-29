import numpy as np
import scipy.spatial as sp

def haus_dist(A,B,b2=None,diam=False):
  asqr = np.sum(A**2, axis=1)
  bsqr = np.sum(B**2, axis=1) if b2 is None else b2
  ab = np.matmul(A, B.T)
  asqr = asqr.reshape(-1, 1)
  dists = np.sqrt(asqr - 2*ab + bsqr)
  haus_d = max(np.max(np.min(dists,axis=1)),np.max(np.min(dists,axis=0)))
  if diam:
    return haus_d,np.max(dists)
  else:
    return haus_d

def scipyHausDist(A, B):
    m = sp.distance.directed_hausdorff(A, B)[0]
    n = sp.distance.directed_hausdorff(B, A)[0]
    return max(m, n)


def fastHD(A, B, T):
    return max(A.transform(T).asymm_dH(B),
                B.transform(T.invert()).asymm_dH(A))