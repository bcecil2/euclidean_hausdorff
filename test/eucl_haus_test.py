import numpy as np
import unittest

from euclidean_hausdorff.eucl_haus import upper_heuristic, upper_exhaustive
from euclidean_hausdorff.transformation import Transformation
from euclidean_hausdorff.point_cloud import PointCloud


class TestEuclHaus(unittest.TestCase):

    def test_box_heuristic_deh(self):
        box = np.array([[1, 1],
                        [-1, 1],
                        [-1, -1],
                        [1, -1]])
        T = Transformation(np.array([1, 2]), [np.pi / 7], False)
        transformed_box = T.apply(box)
        A, B = map(PointCloud, [box, transformed_box])

        dEH, err_ub = upper_heuristic(A.coords, B.coords, p=10)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert np.isclose(0, np.round(dEH, 2))

    def test_box_exact_deh(self):
        box = np.array([[1, 1],
                        [-1, 1],
                        [-1, -1],
                        [1, -1]])
        T = Transformation(np.array([1, 2]), [np.pi / 7], False)
        transformed_box = T.apply(box)
        A, B = map(PointCloud, [box, transformed_box])

        dEH, _ = upper_exhaustive(A.coords, B.coords, target_err=.4)
        assert np.isclose(0, np.round(dEH, 1))

    def test_cube_heuristic_deh(self):
        cube = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 0, 1],
                         [0, 1, 1]])
        T = Transformation(np.array([1, 2, 3]), [np.pi / 7, np.pi / 3, 0], False)
        transformed_cube = T.apply(cube)
        A, B = map(PointCloud, [cube, transformed_cube])

        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, err_ub = upper_heuristic(A.coords, B.coords, p=2)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert dEH < dH

    def test_random_2d_clouds_heuristic(self):
        A_coords = np.random.randn(100, 2)
        T = Transformation(np.array([-1, 2]), [np.pi / 3], True)
        B_coords = T.apply(A_coords)
        A, B = map(PointCloud, [A_coords, B_coords])

        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, err_ub = upper_heuristic(A.coords, B.coords, p=2)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert dEH < dH

    def test_random_3d_clouds_heuristic(self):
        A_coords = np.random.randn(100, 3)
        T = Transformation(np.array([-1, 2, -3]), [np.pi / 3, np.pi / 3, np.pi / 3], True)
        B_coords = T.apply(A_coords)
        A, B = map(PointCloud, [A_coords, B_coords])

        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, err_ub = upper_heuristic(A.coords, B.coords, p=2)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert dEH < dH

if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
