import numpy as np
import unittest
import time
from itertools import starmap

from euclidean_hausdorff import upper, Transformation


def run_perf_test(descr, As_coords, Bs_coords, param_combos):
    for max_n_iter, target_acc, dH_iter_share in param_combos:
        dehs = []
        errs = []
        times = []
        for A_coords, B_coords in zip(As_coords, Bs_coords):
            tic = time.time()
            deh, err_ub = upper(
                A_coords, B_coords, max_n_iter=max_n_iter, target_acc=target_acc,
                dH_iter_share=dH_iter_share)
            toc = time.time()
            dehs.append(deh)
            errs.append(err_ub)
            times.append(toc - tic)

        print(f'[{time.ctime()}] {descr} | {max_n_iter=} | {dH_iter_share=:.1%} '
              f'| {target_acc=}: avg.deh={np.mean(dehs):.4f}, avg.err={np.mean(errs):.4f}, '
              f'avg.time={np.mean(times):.4f}s (total {sum(times)/60:.1f} minutes)')


class PerfEuclHaus2D(unittest.TestCase):
    np.random.seed(0)
    n_shape_size = 20
    n_shape_pairs = 20
    param_combos = [(10, None, .1), (10, None, .2), (100, None, .05),
                    (100, None, .1), (100, None, .2), (1000, None, .05),
                    (1000, None, .1), (1000, None, .2), (0, .01, .05),
                    (0, .01, .1), (0, .01, .2)]

    def test_random_2d_clouds(self):
        As_coords, Bs_coords = np.random.randn(2, self.n_shape_pairs, self.n_shape_size, 2)
        run_perf_test(
            'random 2d clouds', As_coords, Bs_coords, self.param_combos)

    def test_copied_2d_clouds(self):
        As_coords = np.random.randn(self.n_shape_pairs, self.n_shape_size, 2)
        deltas = np.random.randn(self.n_shape_pairs, 2)
        rhos = np.random.randn(self.n_shape_pairs, 1)
        sigmas = np.random.randint(0, 2, self.n_shape_pairs).astype(bool)
        Ts = starmap(Transformation, zip(deltas, rhos, sigmas))
        Bs_coords = [T.apply(A_coords) for T, A_coords in zip(Ts, As_coords)]
        run_perf_test(
            'copied 2d clouds', As_coords, Bs_coords, self.param_combos)


class PerfEuclHaus3D(unittest.TestCase):
    np.random.seed(0)
    n_shape_size = 10
    n_shape_pairs = 10
    param_combos = [(10, None, .1), (10, None, .2), (100, None, .05),
                    (100, None, .1), (100, None, .2), (0, .2, .05),
                    (0, .2, .1), (0, .2, .2)]

    def test_random_3d_clouds(self):
        As_coords, Bs_coords = np.random.randn(2, self.n_shape_pairs, self.n_shape_size, 3)
        run_perf_test(
            'random 3d clouds', As_coords, Bs_coords, self.param_combos)

    def test_copied_3d_clouds(self):
        As_coords = np.random.randn(self.n_shape_pairs, self.n_shape_size, 3)
        deltas = np.random.randn(self.n_shape_pairs, 3)
        rhos = np.random.randn(self.n_shape_pairs, 3)
        sigmas = np.random.randint(0, 2, self.n_shape_pairs)
        Ts = starmap(Transformation, zip(deltas, rhos, sigmas))
        Bs_coords = [T.apply(A_coords) for T, A_coords in zip(Ts, As_coords)]
        run_perf_test(
            'copied 3d clouds', As_coords, Bs_coords, self.param_combos)


if __name__ == "__main__":
    unittest.main()
