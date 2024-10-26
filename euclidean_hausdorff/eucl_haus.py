import numpy as np
from scipy import spatial as sp
from itertools import product, starmap
from sortedcontainers import SortedList

from .point_cloud import PointCloud
from .transformation import Transformation


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


def upper(A_coords, B_coords, max_n_iter=None, target_acc=None, target_err=None,
          dH_iter_share=.1, proper_rigid=False, p=2, verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance using multiscale grid search. The search
    terminates when additive approximation error is ≤ target_acc*max_diam OR when the
    smallest dH found does not improve after max_n_no_improv iterations (depending on
    whether target_acc or max_n_no_improv is set, accordingly)

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param max_n_iter: maximum number of iterations, int
    :param target_acc: target (upper bound of) accuracy as a percentage of larger diameter, float [0, 1]
    :param target_err: target (upper bound of) additive approximation error, float
    :param dH_iter_share: proportion of dH-minimizing iterations, float [0, 1]
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
    assert (not max_n_iter) + (target_acc is None) + (target_err is None), \
        'exactly one of max_n_iter, target_acc, and target_err must be specified'

    # Set maximum iterations if needed.
    max_n_iter = max_n_iter or np.inf

    # Set target error if needed.
    if target_acc is not None:
        max_diam = max(map(diam, [A.coords, B.coords]))
        target_err = target_acc * max_diam
    elif target_err is None:
        target_err = 0

    # Calculate the step at which to perform dH-minimizing iterations.
    dH_iter_step = round(1 / dH_iter_share)

    # Initialize parameters of the multiscale search grid.
    r = np.linalg.norm(normalized_coords, axis=1).max()
    dim_delta, dim_rho = k, k * (k - 1) // 2
    sigmas = [False] if proper_rigid else [False, True]
    eps_delta = np.sqrt(dim_delta)*2*r  # scale-0 cell radius s.t. #∆=1
    eps_rho = eps_delta / ((2*r) if dim_delta == 2 else r)  # adhering to the optimal balance
    a_delta, a_rho = 2*eps_delta / np.sqrt(dim_delta), 2*eps_rho / np.sqrt(dim_rho)    # scale-0 cell sizes

    def calc_dH(delta, rho):    # calculate (smallest) dH for a translation-rotation combo
        dH = np.inf
        for sigma in sigmas:
            T = Transformation(delta, rho, sigma)
            sigma_dH = max(A.transform(T).asymm_dH(B), B.transform(T.invert()).asymm_dH(A))
            dH = min(dH, sigma_dH)
        return dH

    dH_diff_ubs = dict()    # maximum dH discrepancy in a grid cell w.r.t. the cell center

    def calc_dH_diff_ub(i):   # calculate maximum Lipschitz-based dH discrepancy at scale i
        try:
            dH_diff_ub = dH_diff_ubs[i]
        except KeyError:
            diff_delta, diff_rho = np.array([eps_delta, eps_rho]) / p**i
            dH_diff_ub = diff_delta + np.sqrt(2 * (1 - np.cos(diff_rho))) * r
            dH_diff_ubs[i] = dH_diff_ub
        return dH_diff_ub

    def zoom_in(delta, rho, i):   # refine grid cell centered at (δ, ρ) at scale i
        a_delta_i, a_rho_i = np.array([a_delta, a_rho]) / p**i
        deltas, _ = make_grid(delta, a_delta_i / p, 2*r, l=a_delta_i)
        rhos, _ = make_grid(rho, a_rho_i / p, np.pi, l=a_rho_i)
        return deltas, rhos

    # Initialize queue with the multiscale search grid points.
    Qs = [SortedList()]

    def update_grid(deltas, rhos, i, min_found_dH): # process new grid points at scale i
        # Compute dH at each grid point.
        new_points = list(product(map(tuple, deltas), map(tuple, rhos)))
        new_dHs = list(starmap(calc_dH, new_points))

        # Add new grid points to the queue.
        try:
            Q_i = Qs[i]
        except IndexError:
            Q_i = SortedList()
            Qs.append(Q_i)
        Q_i.update(zip(new_dHs, new_points))

        # Update best dH and prune grid points whose cells cannot improve on it.
        # min_found_dH = min(min_found_dH, min(new_dHs))
        min_new_dH = min(new_dHs)
        if min_new_dH < min_found_dH:
            min_found_dH = min_new_dH
            for j, Q_j in enumerate(Qs):
                del Q_j[Q_j.bisect_left((min_found_dH + calc_dH_diff_ub(j),)):]

        # Find grid points with smallest dH and possible dH.
        min_dH = min_possible_dH = np.inf
        min_dH_i = min_possible_dH_i = None
        for j, Q_j in enumerate(Qs):
            if Q_j:
                dH, _ = Q_j[0]
                if dH < min_dH:
                    min_dH = dH
                    min_dH_i = j
                possible_dH = dH - calc_dH_diff_ub(j)
                if possible_dH < min_possible_dH:
                    min_possible_dH = possible_dH
                    min_possible_dH_i = j
        err_ub = max(0, min_found_dH - max(0, min_possible_dH))

        return min_dH_i, min_possible_dH_i, min_found_dH, err_ub

    # Create search grid points of level 0.
    init_deltas, _ = make_grid((0,)*dim_delta, a_delta, 2*r)
    init_rhos, _ = make_grid((0,)*dim_rho, a_rho, np.pi)
    min_dH_i, min_possible_dH_i, min_found_dH, err_ub = update_grid(
        init_deltas, init_rhos, 0, np.inf)

    if verbose > 0:
        print(f'{r=:.5f}, {max_n_iter=}, {dH_iter_step=}, {target_err=:.5f}')

    # Perform multiscale search.
    iter = 0
    while iter < max_n_iter and err_ub > target_err and sum(map(len, Qs)) > 0:
        # Choose the grid cell to refine as having...
        # ...smallest dH, if it's a dH-minimizing iteration.
        if (iter + 1) % dH_iter_step == 0:
            i = min_dH_i
            iter_descr = 'dH-minimizing'
        # ...smallest possible dH, if it's an error-minimizing iteration.
        else:
            i = min_possible_dH_i
            iter_descr = 'error-minimizing'

        # Log the iteration if needed.
        if verbose > 2:
            Q_sizes = {j: len(Q_j) for j, Q_j in enumerate(Qs)}
            dH, _ = Qs[i][0]
            print(f'({iter}: {iter_descr}) {min_found_dH=:.5f}, '
                  f'{err_ub=:.5f}, #Q: {Q_sizes}, zooming in on '
                  f'({i}, {dH:.5f}, {dH - calc_dH_diff_ub(i):.5f})')

        # Refine the chosen grid cell.
        _, (delta, rho) = Qs[i].pop(0)
        new_deltas, new_rhos = zoom_in(delta, rho, i)
        min_dH_i, min_possible_dH_i, min_found_dH, err_ub = update_grid(
            new_deltas, new_rhos, i+1, min_found_dH)

        # Increment iteration index.
        iter += 1

    return min_found_dH, err_ub
