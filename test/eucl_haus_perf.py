import numpy as np
import unittest
import time
from itertools import product

from euclidean_hausdorff import upper, Transformation


class PerfEuclHaus2D(unittest.TestCase):
    np.random.seed(0)
    A_coords = np.random.randn(100, 2)
    T = Transformation(np.array([-1, 2]), [np.pi / 3], True)
    B_coords = T.apply(A_coords)
    B_coords = np.random.randn(100, 2)

    def test_random_2d_clouds(self):
        for n_dH_iter, n_total_iter in product([0, 10, 100], [100, 2000]):
            n_err_ub_iter = n_total_iter - n_dH_iter
            tic = time.time()
            deh, err_ub = upper(
                self.A_coords, self.B_coords, n_dH_iter=n_dH_iter,
                n_err_ub_iter=n_err_ub_iter)
            toc = time.time()
            print(f'2d clouds ({n_dH_iter=}, {n_err_ub_iter=}): {deh=:.4f}, {err_ub=:.4f} '
                  f'({toc-tic:.0f}s)')

class PerfEuclHaus3D(unittest.TestCase):
    np.random.seed(0)
    A_coords = np.random.randn(100, 3)
    T = Transformation(np.array([-1, 2, 3]), [np.pi / 2, -np.pi / 7, np.pi / 5], True)
    B_coords = T.apply(A_coords)

    def test_random_3d_clouds(self):
        for n_dH_iter, n_err_ub_iter in product([1, 10, 100], [0]):
            tic = time.time()
            deh, err_ub = upper(
                self.A_coords, self.B_coords, n_dH_iter=n_dH_iter, n_err_ub_iter=n_err_ub_iter)
            toc = time.time()
            print(f'3d clouds ({n_dH_iter=}, {n_err_ub_iter=}): {deh=:.4f}, {err_ub=:.4f} '
                  f'({toc - tic:.0f}s)')


if __name__ == "__main__":
    unittest.main()
