import numpy as np
import unittest

from euclidean_hausdorff import upper, Transformation, PointCloud


class TestEuclHaus(unittest.TestCase):

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

    def test_box_heuristic(self):
        dEH, _ = upper(self.box, self.transformed_box, max_n_iter=20, dH_iter_share=1)
        assert dEH < .005, f'incorrect dEH {dEH} (should be near 0)'

    def test_box_exact(self):
        target_err = .001
        dEH, err_ub = upper(self.box, self.transformed_box, target_err=target_err)
        assert err_ub <= dEH, f'error bound {err_ub} bigger than dEH {dEH}'
        assert err_ub <= target_err, f'error bound {err_ub} bigger than target_err {target_err}'

    def test_cube_heuristic(self):
        dEH, _ = upper(self.cube, self.transformed_cube, max_n_iter=100, dH_iter_share=1)
        assert dEH < .005, f'incorrect dEH {dEH} (should be near 0)'

    def test_cube_exact(self):
        target_err = .25
        dEH, err_ub = upper(self.cube, self.transformed_cube, target_err=target_err)
        assert err_ub <= dEH, f'error bound {err_ub} bigger than dEH {dEH}'
        assert err_ub <= target_err, f'error bound {err_ub} bigger than target_err {target_err}'

    def test_random_2d_clouds_heuristic(self):
        A, B = map(PointCloud, [self.coords_2d, self.transformed_coords_2d])
        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, _ = upper(self.coords_2d, self.transformed_coords_2d, max_n_iter=10, dH_iter_share=1)
        assert dEH < dH, f'dEH {dEH} is not smaller than dH {dH}'

    def test_random_2d_clouds_exact(self):
        target_err = .01
        dEH, err_ub = upper(self.coords_2d, self.transformed_coords_2d, target_err=target_err)
        assert err_ub <= dEH, f'error bound {err_ub} bigger than dEH {dEH}'
        assert err_ub <= target_err, f'error bound {err_ub} bigger than target_err {target_err}'

    def test_random_3d_clouds_heuristic(self):
        A, B = map(PointCloud, [self.coords_3d, self.transformed_coords_3d])
        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, _ = upper(self.coords_3d, self.transformed_coords_3d, max_n_iter=10, dH_iter_share=1)
        assert dEH < dH, f'dEH {dEH} is not smaller than dH {dH}'


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
