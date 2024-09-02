import numpy as np
import unittest
import time

from euclidean_hausdorff import upper, Transformation


class PerfEuclHaus2D(unittest.TestCase):
    np.random.seed(0)
    A_coords = np.random.randn(100, 2)
    T = Transformation(np.array([-1, 2]), [np.pi / 2], True)
    B_coords = T.apply(A_coords)

    def test_random_2d_clouds_heuristic(self):
        for p in [2, 5]:
            max_n_no_improv = 0
            tic = time.time()
            deh, err_ub = upper(self.A_coords, self.B_coords, p=p, max_n_no_improv=max_n_no_improv)
            toc = time.time()
            print(f'heuristic 2d ({p=}, {max_n_no_improv=}): {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')

    def test_random_2d_clouds_exact(self):
        for p in [2, 5]:
            target_acc = .01
            tic = time.time()
            deh, err_ub = upper(self.A_coords, self.B_coords, target_acc=target_acc, p=p)
            toc = time.time()
            print(f'exact 2d ({p=}, {target_acc=}): {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')


class PerfEuclHaus3D(unittest.TestCase):
    np.random.seed(0)
    A_coords = np.random.randn(100, 3)
    T = Transformation(np.array([-1, 2, 3]), [np.pi / 2, -np.pi / 7, np.pi / 5], True)
    B_coords = T.apply(A_coords)

    def test_random_3d_clouds_heuristic(self):
        p = 2
        max_n_no_improv = 0
        tic = time.time()
        deh, err_ub = upper(self.A_coords, self.B_coords, p=p, max_n_no_improv=max_n_no_improv)
        toc = time.time()
        print(f'heuristic 3d ({p=}, {max_n_no_improv=}): {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')

    def test_random_3d_clouds_exact(self):
        p = 2
        target_acc = .2
        tic = time.time()
        deh, err_ub = upper(self.A_coords, self.B_coords, target_acc=target_acc, p=p)
        toc = time.time()
        print(f'exact 3d ({p=}, {target_acc=}): {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')

if __name__ == "__main__":
    unittest.main()
