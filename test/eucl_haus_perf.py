import numpy as np
import unittest
import time

from euclidean_hausdorff.eucl_haus import upper_exhaustive_heuristic, upper_heuristic, upper_exhaustive, diam
from euclidean_hausdorff.transformation import Transformation
from euclidean_hausdorff.point_cloud import PointCloud


class PerfEuclHaus(unittest.TestCase):
    np.random.seed(0)
    A_coords = np.random.randn(10, 2)
    # T = Transformation(np.array([-1, 2]), [np.pi / 2], True)
    T = Transformation(np.array([0, 0]), np.array([np.pi / 2]), False)
    B_coords = T.apply(A_coords)
    A, B = map(PointCloud, [A_coords, B_coords])
    target_err = .2 * max(map(diam, [A_coords, B_coords]))

    def test_random_2d_clouds_heuristic(self):
        p = 2
        tic = time.time()
        deh, err_ub = upper_heuristic(self.A.coords, self.B.coords, p=p, verbose=3)
        toc = time.time()
        print(f'heuristic ({p=}): {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')

    # def test_random_2d_clouds_exhaustive(self):
    #     tic = time.time()
    #     deh, err_ub = upper_exhaustive(self.A.coords, self.B.coords, self.target_err)
    #     toc = time.time()
    #     print(f'exhaustive: {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')
    #
    # def test_random_2d_clouds_exact(self):
    #     p = 5
    #     tic = time.time()
    #     deh, err_ub = upper_exhaustive_heuristic(self.A.coords, self.B.coords, self.target_err, p=p, verbose=2)
    #     toc = time.time()
    #     print(f'exact ({p=}): {deh=:.4f}, {err_ub=:.4f} ({toc-tic:.0f}s)')


if __name__ == "__main__":
    unittest.main()
