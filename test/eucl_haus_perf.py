import numpy as np
import unittest
import time
from itertools import starmap

from euclidean_hausdorff import upper, Transformation


def run_perf_test(descr, As_coords, Bs_coords, n_dH_iter, n_err_ub_iter):
    dehs = []
    errs = []
    times = []
    for A_coords, B_coords in zip(As_coords, Bs_coords):
        tic = time.time()
        deh, err_ub = upper(
            A_coords, B_coords, n_dH_iter=n_dH_iter, n_err_ub_iter=n_err_ub_iter)
        toc = time.time()
        dehs.append(deh)
        errs.append(err_ub)
        times.append(toc - tic)

    print(f'{descr} ({n_dH_iter=}, {n_err_ub_iter=}): avg. deh={np.mean(dehs):.4f},'
          f' avg. err={np.mean(errs):.4f}, total time={sum(times):.0f}s')


class PerfEuclHaus2D(unittest.TestCase):
    np.random.seed(0)
    n_shape_size = 25
    n_shape_pairs = 10
    n_dH_err_iter_combos = [(0, 10), (10, 0), (100, 0), (0, 100), (10, 1000)]

    def test_random_2d_clouds(self):
        As_coords, Bs_coords = np.random.randn(2, self.n_shape_pairs, self.n_shape_size, 2)
        for n_dH_iter, n_err_ub_iter in self.n_dH_err_iter_combos:
            run_perf_test(
                'random 2d clouds', As_coords, Bs_coords, n_dH_iter, n_err_ub_iter)

    def test_copied_2d_clouds(self):
        As_coords = np.random.randn(self.n_shape_pairs, self.n_shape_size, 2)
        deltas = np.random.randn(self.n_shape_pairs, 2)
        rhos = np.random.randn(self.n_shape_pairs, 1)
        sigmas = np.random.randint(0, 2, self.n_shape_pairs).astype(bool)
        Ts = starmap(Transformation, zip(deltas, rhos, sigmas))
        Bs_coords = [T.apply(A_coords) for T, A_coords in zip(Ts, As_coords)]
        for n_dH_iter, n_err_ub_iter in self.n_dH_err_iter_combos:
            run_perf_test(
                'copied 2d clouds', As_coords, Bs_coords, n_dH_iter, n_err_ub_iter)


class PerfEuclHaus3D(unittest.TestCase):
    np.random.seed(0)
    n_shape_size = 25
    n_shape_pairs = 5
    n_dH_err_iter_combos = [(0, 5), (5, 0), (5, 50)]

    def test_random_2d_clouds(self):
        As_coords, Bs_coords = np.random.randn(2, self.n_shape_pairs, self.n_shape_size, 3)
        for n_dH_iter, n_err_ub_iter in self.n_dH_err_iter_combos:
            run_perf_test(
                'random 3d clouds', As_coords, Bs_coords, n_dH_iter, n_err_ub_iter)

    def test_copied_2d_clouds(self):
        As_coords = np.random.randn(self.n_shape_pairs, self.n_shape_size, 3)
        deltas = np.random.randn(self.n_shape_pairs, 3)
        rhos = np.random.randn(self.n_shape_pairs, 3)
        sigmas = np.random.randint(0, 2, self.n_shape_pairs)
        Ts = starmap(Transformation, zip(deltas, rhos, sigmas))
        Bs_coords = [T.apply(A_coords) for T, A_coords in zip(Ts, As_coords)]
        for n_dH_iter, n_err_ub_iter in self.n_dH_err_iter_combos:
            run_perf_test(
                'copied 3d clouds', As_coords, Bs_coords, n_dH_iter, n_err_ub_iter)


if __name__ == "__main__":
    unittest.main()
