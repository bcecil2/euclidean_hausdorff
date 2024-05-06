from vedo import *
import numpy as np
from glob import glob
from scipy.spatial.distance import cdist
from tqdm import tqdm
from eucl_haus import approx_eucl_haus
from random import shuffle

if __name__ == "__main__":
    anyr = "IntrA/generated/aneurysm/obj/*.obj"
    vessel = "IntrA/generated/vessel/obj/*.obj"
    clouds = [(Mesh(f).vertices, 0) for f in glob(vessel)]
    clouds += [(Mesh(f).vertices, 1) for f in glob(anyr)]
    shuffle(clouds)
    xs = [x[0] for x in clouds]
    ys = [x[1] for x in clouds]
    acc = 0.0
    N = len(xs)
    for i in tqdm(range(N)):
        x = xs[i]
        x /= np.max(cdist(x, x))
        dist,minIdx = min([(approx_eucl_haus(x,y/np.max(cdist(y, y)),max_no_improv=1)[0],j) for j,y in enumerate(xs) if i != j])
        acc += (ys[i] == ys[minIdx])
    acc /= N
    print(f"accuracy: {acc}")



