import numpy as np
import torch
import unittest

from euclidean_hausdorff import upper, optimize_deh_riem, Transformation, PointCloud

class TestEuclHaus(unittest.TestCase):

    np.random.seed(0)
    torch.manual_seed(0)

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

    def test_riem_opt_box_accuracy(self):
        dEH, _, _ = optimize_deh_riem(self.box, self.transformed_box, n_restarts=5, min_dist=1e-6)
        assert dEH < 1e-6, f'riem opt incorrect dEH {dEH} (should be near 0)'

        dEH, _, _ = optimize_deh_riem(self.box, self.transformed_box, n_restarts=5, min_dist=1e-6, joint_opt=True)
        assert dEH < 1e-6, f'join opt incorrect dEH {dEH} (should be near 0)'
    
    def test_riem_opt_cube_accuracy(self):
        dEH, _, _ = optimize_deh_riem(self.cube, self.transformed_cube, n_restarts=5, min_dist=1e-6)
        assert dEH < 1e-6, f'riem opt incorrect dEH {dEH} (should be near 0)'

        # have to bump up restarts to pass with joint optimization
        dEH, _, _ = optimize_deh_riem(self.cube, self.transformed_cube, n_restarts=10, min_dist=1e-6, joint_opt=True)
        assert dEH < 1e-6, f'joint opt incorrect dEH {dEH} (should be near 0)'
    
    def test_riem_opt_hypercube_accuracy(self):
        dim = 4
        cube_coords = np.array(np.meshgrid(*[[0,1]]*dim)).T.reshape(-1,dim)

        theta = np.pi / 4
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([
            [c, -s,  0,  0],
            [s,  c,  0,  0],
            [0,  0,  -1,  0],
            [0,  0,  0,  -1]
        ])
        transformed_cube_coords = cube_coords@rot + [1,2,3,4]
        
        dEH, _, _ = optimize_deh_riem(cube_coords, transformed_cube_coords, n_restarts=5, min_dist=1e-6)
        assert dEH < 1e-6, f'riem opt incorrect dEH {dEH} (should be near 0)'

        dEH, _, _ = optimize_deh_riem(cube_coords, transformed_cube_coords, n_restarts=5, min_dist=1e-6)
        assert dEH < 1e-6, f'joint opt incorrect dEH {dEH} (should be near 0)'
    
    def test_riem_opt_random_min_dist(self):
        A,B = np.random.randn(50,3), np.random.randn(50,3)
        min_dist = 1.5
        dEH, _, _ = optimize_deh_riem(A, B, n_restarts=5, min_dist=min_dist)
        assert dEH <= min_dist, f'riem opt incorrect dEH {dEH} (bigger than min_dist {min_dist})'

        dEH, _, _ = optimize_deh_riem(A, B, n_restarts=5, min_dist=min_dist, joint_opt=True)
        assert dEH <= min_dist, f'joint opt incorrect dEH {dEH} (bigger than min_dist {min_dist})'

    def test_riem_opt_so_path(self):
        A = np.array([[1., 0.], [0., 1.]])
        B = np.array([[0., -1.], [1., 0.]])  # 90 degree rotation

        dEH, o_opt, t_opt = optimize_deh_riem(A, B, n_restarts=3, min_dist=1e-6, special_eucl=True)
        assert dEH < 1e-6, f'SO riem opt incorrect dEH {dEH} (should be near 0)'

        expected_o = np.array([[0., -1.], [1., 0.]])
        assert np.allclose(o_opt, expected_o) or np.allclose(o_opt, -expected_o), f'Incorrect optimal rotation {o_opt}'

        # have to bump up restarts to pass with joint optimization
        dEH, o_opt, t_opt = optimize_deh_riem(A, B, n_restarts=10, min_dist=1e-6, special_eucl=True, joint_opt=True)
        assert dEH < 1e-6, f'joint opt incorrect dEH {dEH} (should be near 0)'

        expected_o = np.array([[0., -1.], [1., 0.]])
        assert np.allclose(o_opt, expected_o,atol=1e-06) or np.allclose(o_opt, -expected_o,atol=1e-06), f'Incorrect optimal rotation {o_opt}'

        

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
