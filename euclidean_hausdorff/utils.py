import matplotlib.pyplot as plt
import numpy as np


def plot(A,B):
  plt.scatter(A[:,0],A[:,1])
  plt.scatter(B[:,0],B[:,1])
  plt.show()


def distort(A,O=np.eye(2),p=np.eye(2),t=np.zeros(2)):
  return A@O@p + t