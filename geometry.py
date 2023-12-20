import numpy as np
from scipy import spatial as sp
from itertools import combinations

def rot2d(theta):
  return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def rot3d(n,theta):
  #page 5 from here https://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
  n1,n2,n3 = n
  c,s = np.cos(theta),np.sin(theta)
  row1 = [c + n1**2*(1-c), n1*n2*(1-c) - n3*s, n1*n3*(1-c) + n2*s]
  row2 = [n1*n2*(1-c) + n3*s, c + n2**2*(1-c), n2*n3*(1-c)-n1*s]
  row3 = [n1*n3*(1-c)-n2*s, n2*n3*(1-c) + n1*s, c + n3**2*(1-c)]
  return np.array([row1,row2,row3])

def reflM(N):
  projMats = []
  idxs = [(i,i) for i in range(N)]
  for i in range(N+1):
    for c in combinations(idxs,i):
      p = np.eye(N)
      for idx in c:
        p[idx] = -1
      projMats.append(p)
  return projMats

def refl2d(nontriv=True):
  R = np.array([[1, 0], [0, -1 if nontriv else 1]], dtype=float)
  return R

def voronoi_bbox(points, bbox):
  '''
  Obtain Voronoi partition of a bounding rectangle.
  '''
  assert points.shape[1] == len(bbox), 'inconsistent dimensionality'

  reflected_points = []
  for dim, bounds in enumerate(bbox):
    coords = points[:, dim]

    assert np.all((bounds[0] <= coords) & (coords <= bounds[1])), 'points outside the bounding box'

    for b in bounds:
      b_reflected_points = np.copy(points)
      b_reflected_points[:, dim] = 2 * b - coords
      reflected_points.extend(b_reflected_points)

  sp_vor = sp.Voronoi(np.concatenate([points, reflected_points]))

  cells = [np.array([sp_vor.vertices[v_idx] for v_idx in sp_vor.regions[p_idx]])
           for p_idx in sp_vor.point_region[:len(points)]]

  return cells
