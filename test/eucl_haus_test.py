import numpy as np
import unittest

from euclidean_hausdorff import upper, Transformation, PointCloud


class TestEuclHaus(unittest.TestCase):

    def test_box_heuristic_deh(self):
        box = np.array([[1, 1],
                        [-1, 1],
                        [-1, -1],
                        [1, -1]])
        T = Transformation(np.array([1, 2]), [np.pi / 7], False)
        transformed_box = T.apply(box)

        dEH, err_ub = upper(box, transformed_box, p=10)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert np.isclose(0, np.round(dEH, 2))

    def test_box_exact_deh(self):
        box = np.array([[1, 1],
                        [-1, 1],
                        [-1, -1],
                        [1, -1]])
        T = Transformation(np.array([1, 2]), [np.pi / 7], False)
        transformed_box = T.apply(box)

        dEH, err_ub = upper(box, transformed_box, target_acc=.001)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
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
        dEH, err_ub = upper(cube, transformed_cube)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert dEH < dH

    def test_random_2d_clouds_heuristic(self):
        A_coords = np.random.randn(100, 2)
        T = Transformation(np.array([-1, 2]), [np.pi / 3], True)
        B_coords = T.apply(A_coords)
        A, B = map(PointCloud, [A_coords, B_coords])
        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, err_ub = upper(A_coords, B_coords)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert dEH < dH

    def test_random_3d_clouds_heuristic(self):
        A_coords = np.random.randn(100, 3)
        T = Transformation(np.array([-1, 2, -3]), [np.pi / 3, np.pi / 3, np.pi / 3], True)
        B_coords = T.apply(A_coords)
        A, B = map(PointCloud, [A_coords, B_coords])
        dH = max(A.asymm_dH(B), B.asymm_dH(A))
        dEH, err_ub = upper(A_coords, B_coords)
        assert np.isclose(dEH, err_ub), f'incorrect error bound {err_ub}'
        assert dEH < dH

    def test_random_2d_clouds_heuristic_err_ub(self):
        A_coords = np.random.randn(100, 2)
        T = Transformation(np.array([-1, 2]), [np.pi / 3], True)
        B_coords = T.apply(A_coords)
        dEH, err_ub = upper(A_coords, B_coords, p=5)
        assert err_ub < .05

if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
