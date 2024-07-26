import numpy as np
import sys
import unittest

sys.path.insert(1, '../euclidean_hausdorff')
from eucl_haus import upper_heuristic, upper_exhaustive
from transformation import Transformation
from point_cloud import PointCloud
from utils import *


np.random.seed(0)
class TestEuclHaus(unittest.TestCase):

    def test_box_heuristic_deh(self):
        box = np.array([[1., 1.],
                        [-1., 1.],
                        [-1., -1.],
                        [1, -1]])
        # this only really works when the rotation is a non standard angle else, the symmetry causes the algorithm
        # to get fooled
        t = Transformation(np.array([1, 2]), [np.pi / 7], False)
        transformed_box = t.apply(box)

        dist, e_ub = upper_heuristic(box, transformed_box,n_parts=10)
        assert np.isclose(0.0,np.round(dist,2))

    def test_box_exact_deh(self):
        box = np.array([[1., 1.],
                        [-1., 1.],
                        [-1., -1.],
                        [1, -1]])
        # this only really works when the rotation is a non standard angle else, the symmetry causes the algorithm
        # to get fooled
        t = Transformation(np.array([1, 2]), [np.pi / 7], False)
        transformed_box = t.apply(box)

        dist, e_ub = upper_exhaustive(box, transformed_box, target_err=0.4)
        assert np.isclose(0.0, np.round(dist, 1))

    def test_cube_heuristic_deh(self):
        cube = np.array([[0., 0., 0.],
                         [1., 0., 0.],
                         [1., 1., 0],
                         [0., 1., 0.],
                         [1., 0, 1.],
                         [1., 1., 1.],
                         [0., 0., 1.],
                         [0., 1., 1.]])
        t = Transformation(np.array([1, 2, 3]), [np.pi / 7, np.pi/3, 0.0], False)
        transformed_cube = t.apply(cube)
        dist, e_ub = upper_heuristic(cube, transformed_cube)
        assert np.isclose(0.0,np.round(dist,2))

    def test_random_clouds_heuristic(self):
        A = np.random.randn(100,3)
        t = Transformation(np.array([-1, 2, -3]), [np.pi / 3, np.pi / 3, np.pi / 3], True)
        B = t.apply(A)
        PA,PB = PointCloud(A),PointCloud(B)
        deh = max(PA.asymm_dH(PB),PB.asymm_dH(PA))
        dist, e_ub = upper_heuristic(A, B, n_parts=3)
        # the random normal data seems to be particularly tricky just make sure were doing better
        # than the original EHD
        assert dist < deh




if __name__ == "__main__":
    unittest.main()
