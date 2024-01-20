import numpy as np
from geometry import rot2d,rot3d
from loss import haus_dist
from tqdm import tqdm
import time
from sklearn.model_selection import GridSearchCV
from classifiers import EuclideanDistClassifier
from sklearn.metrics import make_scorer

def grid_search(A,B,diam,size=50,withTransform=False):
  bsquared = np.sum(B**2, axis=1)
  thetas = np.linspace(0,2*np.pi,size)
  b1s,b2s = np.linspace(-diam,diam,size),np.linspace(-diam,diam,size)
  reflMats = [np.eye(2),np.array([[-1.,0.],[0.,1.]])]
  print("Searching through ", len(thetas)*len(b1s)*len(b2s)*len(reflMats), "elements")
  best = (1e6,None,None,None)
  for t in tqdm(thetas):
    for b1 in b1s:
      for b2 in b2s:
        #outert0 = time.time()
        for refl in reflMats:
          rot = rot2d(t)
          translate = np.array([b1,b2])
          # print(A.shape)
          # print(rot.shape)
          # print(refl.shape)
          T = A@rot@refl + translate
          #t0 = time.time()
          d = haus_dist(T,B,b2=bsquared)
          #t1 = time.time()

          if d < best[0]:
            best = (d,rot,refl,translate)
        #outert1 = time.time()
        #print("Inner loop time: ", outert1 - outert0)
        #print("dist time: ", t1 - t0)
        #print("dist % time: ", (t1-t0)/(outert1-outert0) * 100)
  if withTransform:
    return best
  else:
    return best[0]

def prod(a1,a2):
  return [np.array([a,b]) for a in a1 for b in a2]
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
  return (score,m.rotation,m.reflection,m.shift)

def grid_search3d(A,B,size=50):
  def to_sphere(x,y):
    return (np.sin(x)*np.cos(y),np.sin(x)*np.sin(y),np.cos(x))

  thetas = np.linspace(0,np.pi,size)
  phis = np.linspace(0,2*np.pi,size)
  rhos = np.linspace(0,2*np.pi,size)

  b1s,b2s,b3s = np.linspace(0,1,size),np.linspace(0,1,size),np.linspace(0,1,size)
  reflMats = [np.eye(3),-np.eye(3)]
  best = (1e6,None,None,None)
  p = 1
  for x in [thetas,phis,rhos,b1s,b2s,b3s,reflMats]:
    p *= len(x)
  print("Beginning search through: ", p, " elements")
  for t in thetas:
    for p in phis:
      for r in rhos:
        for b1 in b1s:
          for b2 in b2s:
            for b3 in b3s:
              for refl in reflMats:
                n = to_sphere(t,p)
                rot = rot3d(n,r)
                translate = np.array([b1,b2,b3])
                T = A@rot@refl + translate
                d = haus_dist(T,B)
                if d < best[0]:
                  best = (d,rot,refl,translate)
  return best