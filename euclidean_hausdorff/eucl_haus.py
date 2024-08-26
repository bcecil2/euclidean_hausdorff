import numpy as np
from scipy import optimize
from itertools import product, starmap
from sortedcontainers import SortedList
from tqdm import tqdm

from .point_cloud import PointCloud
from .transformation import Transformation


def make_grid(center, cell_size, ball_rad, cube_size=None):
    """
    Compile a grid with cell size a on the intersection of cube [-l/2, l/2]^k + {c} and ball B(0, r).

    :param center: cube center c, k-array
    :param cell_size: cell side length a, float
    :param ball_rad: ball radius r, float
    :param cube_size: cube side length l, float
    :return: (?, k)-array of grid vertices, cell_size
    """
    # Assume the smallest cube containing the ball if not given.
    cube_size = cube_size or 2*ball_rad

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


def upper_init(A_coords, B_coords, proper_rigid=False, verbose=0):
    """
    Initialize objects and make checks for the Euclidean–Hausdorff distance computation.
    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param verbose: detalization level in the output, int
    :return: function for computing the Hausdorff distance, radius containing the normalized spaces,
        translational dimension, rotational dimension
    """
    # Initialize point clouds.
    A, B = map(PointCloud, [A_coords, B_coords])
    normalized_coords = np.concatenate([A.coords, B.coords])
    _, k = normalized_coords.shape
    assert k in {2, 3}, 'only 2D and 3D spaces are supported'

    # Initialize search grid parameters.
    r = np.linalg.norm(normalized_coords, axis=1).max()
    if verbose:
        print(f'{r=:.5f}')
    dim_delta, dim_rho = k, k * (k - 1) // 2
    sigmas = [False] if proper_rigid else [False, True]

    # Define calculation of the smallest Hausdorff distance under a translation-rotation combo.
    def calc_dH(delta, rho):
        dH = np.inf
        for sigma in sigmas:
            T = Transformation(delta, rho, sigma)
            sigma_dH = max(A.transform(T).asymm_dH(B), B.transform(T.invert()).asymm_dH(A))
            dH = min(dH, sigma_dH)

        return dH

    return calc_dH, r, dim_delta, dim_rho


def upper_exhaustive(A_coords, B_coords, target_err, proper_rigid=False, verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance to the desired error bound
    using exhaustive grid search.

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param target_err: upper bound of additive approximation error, float
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param verbose: detalization level in the output, int
    :return: approximate distance, error upper bound
    """
    calc_dH, r, dim_delta, dim_rho = upper_init(
        A_coords, B_coords, proper_rigid=proper_rigid, verbose=verbose)

    # Find optimal covering radii of the two search grids.
    def obj_grad(eps_delta):
        return (np.arccos(1 - (target_err - eps_delta)**2 / (2 * r**2)) - 2 * eps_delta /
                np.sqrt(4 * r**2 - (target_err - eps_delta)**2))

    eps_delta, = optimize.fsolve(obj_grad, target_err/2)
    eps_rho = np.arccos(1 - (target_err - eps_delta)**2 / (2 * r**2))
    assert np.isclose(target_err, eps_delta + np.sqrt(2*(1 - np.cos(eps_rho)))*r), \
        f"covering radius of translation grid {eps_delta=} was found incorrectly"

    # Make the grid delivering desired error bound.
    delta_center, rho_center = map(np.zeros, [dim_delta, dim_rho])
    deltas, a_delta = make_grid(delta_center, 2*eps_delta / np.sqrt(dim_delta), 2*r)
    rhos, a_rho = make_grid(rho_center, 2*eps_rho / np.sqrt(dim_rho), np.pi)
    err_ub = a_delta * np.sqrt(dim_delta)/2 + a_rho * np.sqrt(dim_rho)/2

    # Exhaustively search the grid.
    delta_rhos = tqdm(product(deltas, rhos), total=len(deltas) * len(rhos), desc="grid vertices")
    best_dH = min(starmap(calc_dH, delta_rhos))

    return best_dH, err_ub


def upper_heuristic(A_coords, B_coords, max_n_restarts=0, improv_margin=.01,
                    proper_rigid=False, p=2, verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance using greedy multiscale grid search.

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param max_n_restarts: limit of restarts (occur when next level yields no improvement), int
    :param improv_margin: relative decrease in dH to count as improvement, float
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param p: number of parts to split a grid cell into (2 means dyadic grid), int
    :param verbose: detalization level in the output, int
    :return: approximate distance, error upper bound
    """
    calc_dH, r, dim_delta, dim_rho = upper_init(
        A_coords, B_coords, proper_rigid=proper_rigid, verbose=verbose)

    # Calculate initial cell sizes/covering radii for ∆ and P.
    a_delta, a_rho = 2*r, 1 if dim_delta == 2 else 2
    eps_delta, eps_rho = a_delta * np.sqrt(dim_delta) / 2, a_rho * np.sqrt(dim_rho) / 2

    def calc_dH_diff_ub(delta_diff, rho_diff):
        return delta_diff + np.sqrt(2 * (1 - np.cos(rho_diff))) * r

    def zoom_in(delta_center, rho_center, level):
        level_a_delta, level_a_rho = np.array([a_delta, a_rho]) / p ** level
        deltas, _ = make_grid(delta_center, level_a_delta / p, 2 * r, cube_size=level_a_delta)
        rhos, _ = make_grid(rho_center, level_a_rho / p, np.pi, cube_size=level_a_rho)
        return deltas, rhos

    # Create a list of sorted (by dH) queues of grid vertices to zoom in on or prune for each level.
    center_delta, center_rho = (0,)*dim_delta, (0,)*dim_rho
    grid_center = (center_delta, center_rho)
    Qs = [SortedList()]
    Qs[0].add((calc_dH(*grid_center), grid_center))
    lvl = 0
    n_restarts = 0

    # Multiscale search until reached the limit of restarts (which happen after
    # not achieving improvement at the next level).
    best_dH = np.inf
    while n_restarts <= max_n_restarts:
        if verbose > 1:
            print(f'{best_dH=:.5f}, {n_restarts=}, '
                  f'Qs={list(map(len, Qs))}, max_dHs={[Q[0][0] for Q in Qs if Q]},'
                  f'Ls={[calc_dH_diff_ub(eps_delta / p ** i, eps_rho / p ** i) for i in range(len(Qs))]}')

        _, (delta, rho) = Qs[lvl].pop(0)

        # Zoom in on the currently best grid vertex.
        child_deltas, child_rhos = zoom_in(delta, rho, lvl)
        children = list(product(map(tuple, child_deltas), map(tuple, child_rhos)))
        child_dHs = list(starmap(calc_dH, children))
        best_child_dH = min(child_dHs)
        try:
            Q = Qs[lvl + 1]
        except IndexError:
            Q = SortedList()
            Qs.append(Q)
        Q.update(zip(child_dHs, children))

        # If some child vertex delivers a non-marginal improvement...
        if best_child_dH < best_dH * (1 - improv_margin):
            lvl += 1
            best_dH = best_child_dH
            n_restarts = 0  # reset the counter of no-improvement iterations

        # If no child vertex is a better candidate to zoom in on...
        else:
            n_restarts += 1  # update the counter of no-improvement iterations
            # Find level of the current best grid vertex to explore next.
            dH = np.inf
            for candidate_lvl in range(len(Qs)):
                if Qs[candidate_lvl]:
                    candidate_dH, _ = Qs[candidate_lvl][0]
                    if candidate_dH < dH:
                        dH, lvl = candidate_dH, candidate_lvl

    # Calculate the error bound based on the maximum possible distance from true optimum
    # to the known grid vertices.
    min_dH_possible = np.inf
    for lvl in range(len(Qs)):
        lvl_err_ub = calc_dH_diff_ub(eps_delta / p**lvl, eps_rho / p**lvl)
        # For each known grid vertex, calculate smallest dH in its "coverage".
        for dH, _ in Qs[lvl]:
            min_dH_in_coverage = max(dH - lvl_err_ub, 0)
            min_dH_possible = min(min_dH_possible, min_dH_in_coverage)

    err_ub = best_dH - min_dH_possible

    return best_dH, err_ub


def upper_exhaustive_heuristic(A_coords, B_coords, target_err, proper_rigid=False, p=2, verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance using greedy (wrt error) multiscale grid search.

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param target_err: upper bound of additive approximation error, float
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param p: number of parts to split a grid cell into (2 means dyadic grid), int
    :param verbose: detalization level in the output, int
    :return: approximate distance, error upper bound
    """
    calc_dH, r, dim_delta, dim_rho = upper_init(
        A_coords, B_coords, proper_rigid=proper_rigid, verbose=verbose)

    # Calculate initial cell sizes/covering radii for ∆ and P.
    a_delta, a_rho = 2*r, 1 if dim_delta == 2 else 2
    eps_delta, eps_rho = a_delta * np.sqrt(dim_delta) / 2, a_rho * np.sqrt(dim_rho) / 2

    def calc_dH_diff_ub(lvl):
        delta_diff, rho_diff = np.array([eps_delta, eps_rho]) / p**lvl
        return delta_diff + np.sqrt(2 * (1 - np.cos(rho_diff))) * r

    def zoom_in(delta_center, rho_center, level):
        level_a_delta, level_a_rho = np.array([a_delta, a_rho]) / p ** level
        deltas, _ = make_grid(delta_center, level_a_delta / p, 2 * r, cube_size=level_a_delta)
        rhos, _ = make_grid(rho_center, level_a_rho / p, np.pi, cube_size=level_a_rho)
        return deltas, rhos

    # Create a list of sorted (by dH) queues of grid vertices to zoom in on or prune for each level.
    center_delta, center_rho = (0,)*dim_delta, (0,)*dim_rho
    grid_center = (center_delta, center_rho)
    lvl = 0
    Q = SortedList()
    Q.add((calc_dH(*grid_center) - calc_dH_diff_ub(lvl), grid_center, lvl))

    # Multiscale search until reached the target error.
    min_found_dH = np.inf
    min_possible_dH = 0   # possible on the unexplored part of the domain
    while min_found_dH - min_possible_dH > target_err:
        if verbose > 1:
            print(f'{min_found_dH=:.5f}, {min_possible_dH=:.5f}')

        min_possible_dH, (delta, rho), lvl = Q.pop(0)

        # Zoom in on the currently best grid vertex.
        child_deltas, child_rhos = zoom_in(delta, rho, lvl)
        children = list(product(map(tuple, child_deltas), map(tuple, child_rhos)))

        child_dHs = np.array(list(starmap(calc_dH, children)))
        min_found_dH = min(min_found_dH, child_dHs.min())
        child_possible_dHs = child_dHs - calc_dH_diff_ub(lvl + 1)
        child_possible_dHs[child_possible_dHs < 0] = 0

        Q.update(zip(child_possible_dHs, children, [lvl + 1] * len(children)))

    return min_found_dH, min_found_dH - min_possible_dH
