import numpy as np
from scipy import spatial as sp
from itertools import product, starmap
from sortedcontainers import SortedList

from .point_cloud import PointCloud
from .transformation import Transformation


DEFAULT_MAX_N_NO_IMPROV = 0
DEFAULT_IMPROV_MARGIN = .01


def diam(coords):
    hull = sp.ConvexHull(coords)
    hull_coords = coords[hull.vertices]
    candidate_distances = sp.distance.cdist(hull_coords, hull_coords)

    return candidate_distances.max()


def make_grid(center, h, r, l=None):
    """
    Compile a grid with cell size h covering the intersection of
    the cube [-l/2, l/2]^k + {c} and ball B(0, r).

    :param center: cube center c, k-array
    :param h: side length of a grid cell, float
    :param r: ball radius, float
    :param l: side length of the cube, float
    :return: (?, k)-array of grid points, updated a (for divisibility)
    """
    # Assume the smallest cube containing the ball if not given.
    l = l or 2 * r

    # Reduce cell size without increasing the cell count.
    n_cells = int(np.ceil(l / h))
    h = l / n_cells

    # Calculate covering radius.
    k = len(center)
    covering_rad = np.sqrt(k) * h / 2

    # Calculate grid point positions separately in each dimension.
    offsets_from_center = np.linspace(-(l - h) / 2, (l - h) / 2, n_cells)
    positions = np.add.outer(center, offsets_from_center)

    # Compile grid point coordinates.
    k = len(positions)
    coords = np.reshape(np.meshgrid(*positions), (k, -1)).T

    # Retain only the grid points covering the ball.
    lengths = np.linalg.norm(coords, axis=1)
    is_covering = lengths <= r + covering_rad
    coords = coords[is_covering]
    lengths = lengths[is_covering]

    # Project grid points outside of the ball onto the ball.
    is_outside = lengths > r
    coords[is_outside] /= lengths[is_outside][:, None]

    return coords, h


def upper(A_coords, B_coords, target_acc=None, target_err=None, max_n_no_improv=None,
          improv_margin=None, proper_rigid=False, p=2, verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance using multiscale grid search. The search
    terminates when additive approximation error is ≤ target_acc*max_diam OR when the
    smallest dH found does not improve after max_n_no_improv iterations (depending on
    whether target_acc or max_n_no_improv is set, accordingly)

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param target_acc: target (upper bound of) accuracy as a percentage of larger diameter, float
    :param target_err: target (upper bound of) additive approximation error, float
    :param max_n_no_improv: max number of steps with no improvement of min dH, int
    :param improv_margin: relative decrease in dH to count as improvement, float
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param p: number of parts to split a grid cell into (e.g. 2 for dyadic), int
    :param verbose: detalization level in the output, int
    :return: approximate dEH, upper bound of additive approximation error
    """
    # Initialize point clouds.
    A, B = map(PointCloud, [A_coords, B_coords])
    normalized_coords = np.concatenate([A.coords, B.coords])
    _, k = normalized_coords.shape

    # Check parameter correctness.
    assert k in {2, 3}, 'only 2D and 3D spaces are supported'
    assert target_acc is None or target_err is None, \
        'only one of target_acc and target_err can be specified'
    is_exact = target_acc is not None or target_err is not None
    if is_exact:
        assert max_n_no_improv is None, \
            'max_n_no_improv cannot be used together with target accuracy/error'
        assert improv_margin is None, \
            'improv_margin can not be used together with target accuracy/error'
    else:
        if max_n_no_improv is None:
            max_n_no_improv = DEFAULT_MAX_N_NO_IMPROV
        if improv_margin is None:
            improv_margin = DEFAULT_IMPROV_MARGIN

    # Initialize parameters of the multiscale search grid.
    r = np.linalg.norm(normalized_coords, axis=1).max()
    dim_delta, dim_rho = k, k * (k - 1) // 2
    sigmas = [False] if proper_rigid else [False, True]
    eps_delta, eps_rho = 2*r, 1 if dim_delta == 2 else 2    # level-0 covering radii (s.t. ∆ has 1 point)
    h_delta, h_rho = 2*eps_delta/np.sqrt(dim_delta), 2*eps_rho/np.sqrt(dim_rho) # level-0 cell sizes

    # Initialize queue with the multiscale search grid points.
    Q = SortedList()

    # Determine search type and approximation error bound (if relevant).
    if target_acc is not None:
        is_exact = True
        max_diam = max(map(diam, [A.coords, B.coords]))
        target_err = target_acc * max_diam
        if verbose > 0:
            print(f'{r=:.4f}, {target_err=:.4f}')
    else:
        is_exact = False
        if verbose > 0:
            print(f'{r=:.5f}, {max_n_no_improv=}')

    # Define functions for multiscale search.
    def calc_dH(delta, rho):    # calculate (smallest) dH for a translation-rotation combo
        dH = np.inf
        for sigma in sigmas:
            T = Transformation(delta, rho, sigma)
            sigma_dH = max(A.transform(T).asymm_dH(B), B.transform(T.invert()).asymm_dH(A))
            dH = min(dH, sigma_dH)
        return dH

    def calc_dH_diff_ub(lvl):   # calculate Lipschitz-based bound for dH discrepancy
        delta_diff, rho_diff = np.array([eps_delta, eps_rho]) / p**lvl
        return delta_diff + np.sqrt(2 * (1 - np.cos(rho_diff))) * r

    def zoom_in(delta_center, rho_center, lvl):   # produce next-level offsprings of a grid point
        lvl_h_delta, lvl_h_rho = np.array([h_delta, h_rho]) / p**lvl
        deltas, _ = make_grid(delta_center, lvl_h_delta / p, 2*r, l=lvl_h_delta)
        rhos, _ = make_grid(rho_center, lvl_h_rho / p, np.pi, l=lvl_h_rho)
        return deltas, rhos

    def process_grid_point_coords(deltas, rhos):
        grid_point_coords = list(product(map(tuple, deltas), map(tuple, rhos)))
        dHs = np.array(list(starmap(calc_dH, grid_point_coords)))
        return dHs, grid_point_coords

    if is_exact:    # storing min dH possible in the covering area of the grid point in the queue
        def update_grid(deltas, rhos, lvl, min_found_dH, n_no_improv):
            dHs, grid_points = process_grid_point_coords(deltas, rhos)
            possible_dHs = dHs - calc_dH_diff_ub(lvl)
            possible_dHs[possible_dHs < 0] = 0
            Q.update(zip(possible_dHs, grid_points, [lvl] * len(grid_points)))
            min_possible_dH, *_ = Q[0]
            min_found_dH = min(min_found_dH, dHs.min())
            return min_found_dH, min_possible_dH, None

        def check_termination(min_found_dH, min_possible_dH, n_no_improv):
            return min_found_dH - min_possible_dH <= target_err

    else:   # storing dH at the grid point in the queue
        def update_grid(deltas, rhos, lvl, min_found_dH, n_no_improv):
            dHs, grid_points = process_grid_point_coords(deltas, rhos)
            Q.update(zip(dHs, grid_points, [lvl] * len(grid_points)))
            min_dH = dHs.min()
            if min_dH >= min_found_dH * (1 - improv_margin):
                n_no_improv += 1    # no improvement beyond marginal
            min_found_dH = min(min_found_dH, min_dH)
            return min_found_dH, None, n_no_improv

        def check_termination(min_found_dH, min_possible_dH, n_no_improv):
            return n_no_improv > max_n_no_improv

    # Create search grid points of level 0.
    init_deltas, _ = make_grid((0,)*dim_delta, h_delta, 2*r)
    init_rhos, _ = make_grid((0,)*dim_rho, h_rho, np.pi)
    min_found_dH, min_possible_dH, n_no_improv = update_grid(
        init_deltas, init_rhos, 0, np.inf, 0)

    # Multiscale search until reached termination condition.
    while not check_termination(min_found_dH, min_possible_dH, n_no_improv):
        if verbose > 1:
            print(f'{min_found_dH=:.5f}, {min_possible_dH=:.5f}, {len(Q)=}')
            if verbose > 2:
                    grid_points = [(rho, delta) for _, (rho, delta), _ in Q]
                    grid_points = [([round(x, 2) for x in rho], [round(x, 2) for x in delta])
                                     for rho, delta in grid_points]
                    print(f'{grid_points}')

        _, (delta, rho), lvl = Q.pop(0)

        # Zoom in on the currently best grid point.
        child_deltas, child_rhos = zoom_in(delta, rho, lvl)
        min_found_dH, min_possible_dH, n_no_improv = update_grid(
            child_deltas, child_rhos, lvl + 1, min_found_dH, n_no_improv)

    # Find min possible dH if the search was not exact.
    if not is_exact:
        min_possible_dH = max(0, min(dH - calc_dH_diff_ub(lvl) for dH, _, lvl in Q))

    return min_found_dH, min_found_dH - min_possible_dH
