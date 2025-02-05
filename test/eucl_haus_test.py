import numpy as np
import unittest

from euclidean_hausdorff import upper, Transformation, PointCloud


class TestEuclHaus(unittest.TestCase):

    np.random.seed(0)

    box = np.array([[1, 1],
                         [-1, 1],
                         [-1, -1],
                         [1, -1]])
    transformed_box = Transformation([1, 2], [np.pi / 7], True).apply(box)

    cube = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 0, 1],
                     [0, 1, 1]])
    transformed_cube = Transformation([1, 2, 3], [np.pi / 7, np.pi / 3, 0], False).apply(cube)

    coords_2d = np.random.randn(100, 2)
    transformed_coords_2d = Transformation(
        [-1, 2], [np.pi / 3], True).apply(coords_2d)

    coords_3d = np.random.randn(100, 3)
    transformed_coords_3d = Transformation(
        [-1, 2, -3], [np.pi / 3, np.pi / 3, np.pi / 3], True).apply(coords_3d)

    def test_box_iter_budget(self):
        dEH, _ = upper(self.box, self.transformed_box, n_err_ub_iter=20, return_err=True)
        assert dEH < .005, f'incorrect dEH {dEH} (should be near 0)'

    def test_box_target_err(self):
        target_err = .001
        dEH, err_ub = upper(self.box, self.transformed_box, target_err=target_err, return_err=True)
        assert err_ub <= dEH, f'error bound {err_ub} bigger than dEH {dEH}'
        assert err_ub <= target_err, f'error bound {err_ub} bigger than target_err {target_err}'

    def test_cube_iter_budget(self):
        dEH, _ = upper(self.cube, self.transformed_cube, n_err_ub_iter=100, return_err=True)
        assert dEH < .01, f'incorrect dEH {dEH} (should be near 0)'

    def test_cube_target_err(self):
        target_err = .25
        dEH, err_ub = upper(self.cube, self.transformed_cube, target_err=target_err, return_err=True)
        assert err_ub <= dEH, f'error bound {err_ub} bigger than dEH {dEH}'
        assert err_ub <= target_err, f'error bound {err_ub} bigger than target_err {target_err}'

    def test_random_2d_clouds_iter_budget(self):
        A, B = map(PointCloud, [self.coords_2d, self.transformed_coords_2d])
        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, _ = upper(self.coords_2d, self.transformed_coords_2d, n_err_ub_iter=20, return_err=True)
        assert dEH < dH, f'dEH {dEH}                                                                                                    is not smaller than dH {dH}'

    def test_random_2d_clouds_iter_budget_smooth(self):
        A, B = map(PointCloud, [self.coords_2d, self.transformed_coords_2d])
        agg = np.mean
        dH = max(A.asymm_dH(B, agg=agg), B.asymm_dH(A, agg=agg))
        dEH, _ = upper(self.coords_2d, self.transformed_coords_2d,
                       n_err_ub_iter=20, return_err=True, agg=agg)
        assert dEH < dH, f'dEH {dEH} is not smaller than dH {dH}'

    def test_random_2d_clouds_target_err(self):
        target_err = .01
        dEH, err_ub = upper(self.coords_2d, self.transformed_coords_2d,
                            target_err=target_err, return_err=True)
        assert err_ub <= dEH, f'error bound {err_ub} bigger than dEH {dEH}'
        assert err_ub <= target_err, f'error bound {err_ub} bigger than target_err {target_err}'

    def test_random_3d_clouds_iter_budget(self):
        A, B = map(PointCloud, [self.coords_3d, self.transformed_coords_3d])
        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, _ = upper(self.coords_3d, self.transformed_coords_3d,
                       n_err_ub_iter=20, return_err=True)
        assert dEH < dH, f'dEH {dEH} is not smaller than dH {dH}'

    def test_random_3d_clouds_iter_budget_smooth(self):
        A, B = map(PointCloud, [self.coords_3d, self.transformed_coords_3d])
        agg = np.mean
        dH = max(A.asymm_dH(B, agg=agg), B.asymm_dH(A, agg=agg))
        dEH, _ = upper(self.coords_3d, self.transformed_coords_3d,
                       n_err_ub_iter=20, return_err=True, agg=agg)
        assert dEH < dH, f'dEH {dEH} is not smaller than dH {dH}'

if __name__ == "__main__":
    unittest.main()
