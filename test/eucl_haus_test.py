import numpy as np
import unittest
import time

from euclidean_hausdorff.eucl_haus import upper_exhaustive_heuristic, upper_heuristic, upper_exhaustive
from euclidean_hausdorff.transformation import Transformation
from euclidean_hausdorff.point_cloud import PointCloud


class TestEuclHaus(unittest.TestCase):

    # def test_box_heuristic_deh(self):
    #     box = np.array([[1, 1],
    #                     [-1, 1],
    #                     [-1, -1],
    #                     [1, -1]])
    #     T = Transformation(np.array([1, 2]), [np.pi / 7], False)
    #     transformed_box = T.apply(box)
    #
    #     dEH, err_ub = upper_heuristic(box, transformed_box, p=10)
    #     assert np.isclose(0, np.round(dEH, 2))
    #
    # def test_box_exact_deh(self):
    #     box = np.array([[1, 1],
    #                     [-1, 1],
    #                     [-1, -1],
    #                     [1, -1]])
    #     T = Transformation(np.array([1, 2]), [np.pi / 7], False)
    #     transformed_box = T.apply(box)
    #
    #     dEH, err_ub = upper_exhaustive_heuristic(box, transformed_box, target_err=.01, p=10)
    #     assert np.isclose(0, np.round(dEH, 2))
    #
    # def test_cube_heuristic_deh(self):
    #     cube = np.array([[0, 0, 0],
    #                      [1, 0, 0],
    #                      [1, 1, 0],
    #                      [0, 1, 0],
    #                      [1, 0, 1],
    #                      [1, 1, 1],
    #                      [0, 0, 1],
    #                      [0, 1, 1]])
    #     T = Transformation(np.array([1, 2, 3]), [np.pi / 7, np.pi / 3, 0], False)
    #     transformed_cube = T.apply(cube)
    #     dEH, _ = upper_heuristic(cube, transformed_cube)
    #     assert np.isclose(0, np.round(dEH, 2))
    #
    # # def test_cube_exact_deh(self):
    # #     cube = np.array([[0, 0, 0],
    # #                      [1, 0, 0],
    # #                      [1, 1, 0],
    # #                      [0, 1, 0],
    # #                      [1, 0, 1],
    # #                      [1, 1, 1],
    # #                      [0, 0, 1],
    # #                      [0, 1, 1]])
    # #     T = Transformation(np.array([1, 2, 3]), [np.pi / 7, np.pi / 3, 0], False)
    # #     transformed_cube = T.apply(cube)
    # #     dEH, _ = upper_exhaustive_heuristic(cube, transformed_cube, target_err=.005)
    # #     assert np.isclose(0, np.round(dEH, 2))
    #
    # def test_random_3d_clouds_heuristic(self):
    #     A_coords = np.random.randn(100, 3)
    #     T = Transformation(np.array([-1, 2, -3]), [np.pi / 3, np.pi / 3, np.pi / 3], True)
    #     B_coords = T.apply(A_coords)
    #     A, B = map(PointCloud, [A_coords, B_coords])
    #     dH = max(A.asymm_dH(B), B.asymm_dH(A))
    #     dEH, _ = upper_heuristic(A_coords, B_coords, p=3)
    #     assert dEH < dH
    #
    def test_random_2d_clouds_heuristic_err_ub(self):
        A_coords = np.random.randn(100, 2)
        T = Transformation(np.array([-1, 2]), [np.pi / 3], True)
        B_coords = T.apply(A_coords)
        deh, err_ub = upper_heuristic(A_coords, B_coords, p=10)
        print(f'{deh=:.4f}, {err_ub=:.4f}')
        assert err_ub < .52


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
