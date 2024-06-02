import numpy as np
from scipy import optimize
from itertools import product
from sortedcontainers import SortedList

from point_cloud import PointCloud
from transformation import Transformation
from optimizers import diam


def make_grid(center, cell_size, cube_size, ball_rad):
    """
    Compile a grid with cell size a on the intersection of cube [-l/2, l/2]^k + {c} and ball B(0, r).

    :param center: cube center c, k-array
    :param cell_size: cell side length a, float
    :param cube_size: cube side length l, float
    :param ball_rad: ball radius r, float
    :return: (?, k)-array of grid vertices, cell_size
    """
    # Reduce cell size without increasing the cell count.
    n_cells = int(np.ceil(cube_size / cell_size))
    cell_size = cube_size / n_cells

    # Calculate covering radius.
    k = len(center)
    covering_rad = np.sqrt(k) * cell_size / 2

    # Calculate vertex positions separately in each dimension.
    vert_offsets = np.linspace(-(cube_size-cell_size)/2, (cube_size-cell_size)/2, n_cells)
    vert_positions = np.add.outer(center, vert_offsets)

    # Generate vertex coordinates.
    k = len(vert_positions)
    vertex_coords = np.reshape(np.meshgrid(*vert_positions), (k, -1)).T

    # Retain only the vertices covering the ball.
    lengths = np.linalg.norm(vertex_coords, axis=1)
    is_covering = lengths <= ball_rad + covering_rad
    vertex_coords = vertex_coords[is_covering]
    lengths = lengths[is_covering]

    # Project vertices outside of the ball onto the ball.
    is_outside = lengths > ball_rad
    vertex_coords[is_outside] /= lengths[is_outside][:, None]

    return vertex_coords, cell_size


def approx_eucl_haus(A_coords, B_coords, target_err=None, max_no_improv=0, improv_margin=.01,
                     proper_rigid=False, distance_agg='max', verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance.

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param target_err: (upper bound of) additive approximation error, float
    :param max_no_improv: maximum number of iterations without improvement, int
    :param improv_margin: relative
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param verbose: detalization level in the output, int
    :return: approximate distance
    """
    A, B = PointCloud(A_coords, distance_agg=distance_agg), PointCloud(B_coords, distance_agg=distance_agg)
    normalized_coords = np.concatenate([A.coords, B.coords])

    _, k = normalized_coords.shape
    assert k in {2, 3}, 'only 2D and 3D spaces are supported'
    r = np.linalg.norm(normalized_coords, axis=1).max()
    if verbose:
        print(f'{r=:.5f}, max diam={max(map(lambda x: diam(x.coords), [A, B])):.5f}')

    # Calculate initial cell sizes/covering radii for ∆ and P .
    a_delta, a_rho = 2*r, 2
    eps_delta, eps_rho = np.array([a_delta, a_rho]) * np.sqrt(k) / 2

    def calc_dH_diff_ub(delta_diff, rho_diff):
        return delta_diff + np.sqrt(2 * (1 - np.cos(rho_diff))) * r

    def zoom_in(point, level):
        delta_center, rho_center = point[:k], point[k:]
        delta_cell_size, rho_cell_size = np.array([a_delta, a_rho]) / 2**level
        deltas, _ = make_grid(delta_center, delta_cell_size/2, delta_cell_size, 2*r)
        rhos, _ = make_grid(rho_center, rho_cell_size/2, rho_cell_size, np.pi)
        delta_part = np.tile(deltas, (len(rhos), 1))
        rho_part = np.repeat(rhos, len(deltas), axis=0)
        return np.hstack((delta_part, rho_part))

    sigmas = [False] if proper_rigid else [False, True]
    best_dH = err_ub = np.inf

    def calc_dH(grid_point):
        dH = np.inf
        for sigma in sigmas:
            T = Transformation(grid_point[:k], grid_point[k:], sigma)
            sigma_dH = max(A.transform(T).asymm_dH(B), B.transform(T.invert()).asymm_dH(A))
            dH = min(dH, sigma_dH)

        return dH

    # Create a list of sorted (by dH) queues of grid points to zoom in on or prune for each level.
    grid_center = np.zeros(k + k * (k-1) // 2)
    Qs = [SortedList()]
    Qs[0].add((calc_dH(grid_center), tuple(grid_center)))
    lvl = min_unexpl_lvl = 0
    n_no_improv = 0
    # Multiscale search until achieved the target accuracy (or searched the maximum number of
    # grid points without improvement in a row, if the target accuracy is not set).
    while ((target_err and err_ub > target_err) or
           (not target_err and n_no_improv <= max_no_improv)):
        if verbose > 1:
            print(f'{best_dH=:.5f}, err_ub={err_ub:.5f}, {n_no_improv=}, '
                  f'Qs={list(map(len, Qs))}')

        _, grid_point = Qs[lvl].pop(0)

        # Zoom in on the currently best grid point.
        children = zoom_in(np.array(grid_point), lvl)
        child_dHs = list(map(calc_dH, children))
        best_child_dH = min(child_dHs)
        try:
            Q = Qs[lvl + 1]
        except IndexError:
            Q = SortedList()
            Qs.append(Q)
        Q.update(zip(child_dHs, map(tuple, children)))

        # If some child point delivers a non-marginal improvement...
        if best_child_dH < best_dH * (1 - improv_margin):
            lvl += 1
            best_dH = best_child_dH
            err_ub = min(best_dH, err_ub)
            n_no_improv = 0  # reset the counter of no-improvement iterations

            # Prune grid points zooming in on which cannot improve best dH.
            for prune_lvl in range(min_unexpl_lvl, len(Qs)):
                prune_lvl_err_ub = calc_dH_diff_ub(eps_delta / 2**prune_lvl, eps_rho / 2**prune_lvl)
                prune_thresh = best_dH + prune_lvl_err_ub
                n_points_to_retain = Qs[prune_lvl].bisect_left((prune_thresh, ))
                for _ in range(len(Qs[prune_lvl]) - n_points_to_retain):
                    Qs[prune_lvl].pop()

        # If no child point is a better candidate to zoom in on...
        else:
            n_no_improv += 1  # update the counter of no-improvement iterations

            # Choose the current best grid point to explore next.
            # lvl = min_unexpl_lvl
            dH = np.inf
            for candidate_lvl in range(min_unexpl_lvl, len(Qs)):
                try:
                    candidate_dH, _ = Qs[candidate_lvl][0]
                except IndexError:
                    pass
                else:
                    if candidate_dH < dH:
                        dH, lvl = candidate_dH, candidate_lvl

        # Update the smallest unexplored level and the associated error bound.
        while not Qs[min_unexpl_lvl]:
            min_unexpl_lvl += 1
            lvl = max(lvl, min_unexpl_lvl)
            min_unexpl_lvl_err_ub = calc_dH_diff_ub(
                eps_delta / 2**min_unexpl_lvl, eps_rho / 2**min_unexpl_lvl)
            err_ub = min(min_unexpl_lvl_err_ub, err_ub)
            if verbose:
                print(f'updated depth range to {min_unexpl_lvl}-{len(Qs) - 1}')

    return best_dH, err_ub
