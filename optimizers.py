import numpy as np
from geometry import rot2d,rot3d
from loss import haus_dist,fastHD
from tqdm import tqdm
from time import perf_counter
from sklearn.model_selection import GridSearchCV
from classifiers import EuclideanDistClassifier
from sklearn.metrics import make_scorer
from scipy import spatial as sp
from point_cloud import PointCloud
from transformation import Transformation
from itertools import product


def to_sphere(x, y):
  return (np.sin(x) * np.cos(y), np.sin(x) * np.sin(y), np.cos(x))

def cartesian_product(*arrays):
  la = len(arrays)
  dtype = np.result_type(*arrays)
  arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
  for i, a in enumerate(np.ix_(*arrays)):
    arr[..., i] = a
  return arr.reshape(-1, la)

def diam(coords):
  hull = sp.ConvexHull(coords)
  hull_coords = coords[hull.vertices]
  candidate_distances = sp.distance.cdist(hull_coords, hull_coords)
  return candidate_distances.max()

def grid_search(A,B,grid_size=50):
  A, B = PointCloud(A),PointCloud(B)
  max_shift = max(diam(A.coords), diam(B.coords))
  xshifts = np.linspace(-max_shift, max_shift, grid_size)
  yshifts = np.linspace(-max_shift, max_shift, grid_size)
  thetas = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
  def hd(p):
    T = Transformation([p[0], p[1]], [p[2]], p[3])
    return fastHD(A, B, T)
  f = np.vectorize(hd,signature='(4)->()')
  grid = cartesian_product(xshifts,yshifts,thetas,np.array([0,1]))
  return f(grid).min()



def grid_search_3d(A,B,grid_size=5):
  A, B = PointCloud(A), PointCloud(B)

  max_shift = max(diam(A.coords), diam(B.coords))
  xshifts = np.linspace(-max_shift, max_shift, grid_size)
  yshifts = np.linspace(-max_shift, max_shift, grid_size)
  zshifts = np.linspace(-max_shift, max_shift, grid_size)
  thetas = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
  phis = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
  rhos = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)

  def hd(p):
    n = to_sphere(p[0], p[1])
    theta = p[2]
    translate = [p[3],p[4],p[5]]
    reflect = p[6]
    T = Transformation(translate, [n,theta], reflect)
    return fastHD(A, B, T)
  f = np.vectorize(hd, signature='(7)->()')
  grid = cartesian_product(xshifts, yshifts, zshifts, thetas, phis, rhos, np.array([0, 1]))
  return f(grid).min()

def grid_search_sklearn(A,B,diam,size=50):
  bsquared = np.sum(B**2, axis=1)
  t0 = time.time()
  rots = [rot2d(t) for t in  np.linspace(0,2*np.pi,size)]
  shifts = prod(np.linspace(-diam,diam,size),np.linspace(-diam,diam,size))
  reflMats = [np.eye(2),np.array([[-1.,0.],[0.,1.]])]
  t1 = time.time()
  print("Parallel GS setup time: ", t1-t0)
  #print("Searching through ", len(thetas)*len(b1s)*len(b2s)*len(reflMats), "elements")
  grid = {"rotation":rots,
          "reflection":reflMats,
          "shift":shifts}
  model = EuclideanDistClassifier()
  model.fit(A,B)
  scorer = make_scorer(haus_dist, greater_is_better=False)
  searcher = GridSearchCV(estimator=model,param_grid=grid,scoring=scorer, cv=( ((slice(None), slice(None)), )), n_jobs=4, verbose=3)
  searcher.fit(A,B)
  m = searcher.best_estimator_
  score = searcher.best_score_
  return (score, m.rotation, m.do_reflect, m.translate)


