import keras
import tensorflow
from euclidean_hausdorff import upper
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import  tqdm

def im2points(X):
    xs,ys = np.nonzero(X)
    points = np.stack([xs,ys],axis=-1).astype(np.float64)
    points /= np.max(cdist(points,points))
    return points

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


    xs = x_test[:20]
    xs = [im2points(x) for x in xs]
    # once a target err is fixed exhaustive is the same just grab it once first
    _, _, nc, num_exhaustive_evals = upper(xs[0], xs[1], n_dH_iter=0, target_err=1)
    num_error_min_evals = []
    for i in tqdm(range(len(xs))):
         for j in range(len(xs)):
             if i != j:
                _, _, nc1, _ = upper(xs[i], xs[j], n_dH_iter=0, target_err=1)
                num_error_min_evals.append(nc1)
    num_error_min_evals = np.array(num_error_min_evals)
    num_exhaustive_evalations = np.array([num_exhaustive_evals])
    np.save("n_err_evals.npy", num_error_min_evals)
    np.save("n_exh_evals.npy", num_exhaustive_evals)



