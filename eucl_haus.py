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


def approx_eucl_haus(A_coords, B_coords, alpha, proper_rigid=False, verbose=0):
    """
    Approximate the Euclidean–Hausdorff distance.

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param alpha: (upper bound of) additive approximation error, float
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :param verbose: detalization level in the output, int
    :return: approximate distance
    """
    A, B = PointCloud(A_coords), PointCloud(B_coords)
    normalized_coords = np.concatenate([A.coords, B.coords])

    _, k = normalized_coords.shape
    assert k in {2, 3}, 'only 2D and 3D spaces are supported'
    r = np.linalg.norm(normalized_coords, axis=1).max()

    # Calculate initial cell sizes/covering radii for ∆ and P .
    a_delta, a_rho = 2*r, 2
    eps_delta, eps_rho = np.array([a_delta, a_rho]) * np.sqrt(k) / 2

    def dH_diff_ub(delta_diff, rho_diff):
        return delta_diff + np.sqrt(2 * (1 - np.cos(rho_diff))) * r

    # Calculate maximum dyadic grid depth m.
    m = 1
    while dH_diff_ub(eps_delta / 2**m, eps_rho / 2**m) > alpha:
        m += 1

    if verbose:
        print(f'{r=:.5f}, max diam={max(map(lambda x: diam(x.coords), [A, B])):.5f}')
        for l in range(m):
            print(f'level {l}, pruning offset {dH_diff_ub(eps_delta / 2**l, eps_rho / 2**l):.5f}')

    def zoom_in(point, level):
        delta_center, rho_center = point[:k], point[k:]
        delta_cell_size, rho_cell_size = np.array([a_delta, a_rho]) / 2**level
        deltas, _ = make_grid(delta_center, delta_cell_size/2, delta_cell_size, 2*r)
        rhos, _ = make_grid(rho_center, rho_cell_size/2, rho_cell_size, np.pi)
        delta_part = np.tile(deltas, (len(rhos), 1))
        rho_part = np.repeat(rhos, len(deltas), axis=0)
        return np.hstack((delta_part, rho_part))

    best_dH = np.inf
    sigmas = [False] if proper_rigid else [False, True]
    for sigma in sigmas:
        def dH(grid_point):
            T = Transformation(grid_point[:k], grid_point[k:], sigma)
            return max(A.transform(T).asymm_dH(B),
                       B.transform(T.invert()).asymm_dH(A))

        # Create a sorted (by dH) queue of grid points to zoom in on or prune for each level.
        Qs = [SortedList() for _ in range(m)]
        grid_center = np.zeros(k + k * (k-1) // 2)
        Qs[0].add((dH(grid_center), tuple(grid_center)))
        level = shallowest_level = 0
        # Multiscale search until all points of level < m are zoomed in on or pruned,
        # or the desired accuracy is trivially achieved.
        while sum(map(len, Qs[shallowest_level:])) > 0 and best_dH > alpha:
            if verbose > 1:
                print(f'{best_dH=:.5f}, Qs={list(map(len, Qs))}')

            _, grid_point = Qs[level].pop(0)
            if not Qs[level] and level == shallowest_level:
                shallowest_level += 1
                if verbose:
                    print(f'{shallowest_level=}/{m}')

            # Zoom in on the currently best grid point.
            children = zoom_in(np.array(grid_point), level)
            level += 1
            children_dH = list(map(dH, children))
            best_child_dH = min(children_dH)
            if level < m:
                Qs[level].update(zip(children_dH, map(tuple, children)))

            if best_child_dH < best_dH:
                # Prune grid points zooming in on which cannot improve best dH.
                for l in range(shallowest_level, m):
                    dH_thresh = best_child_dH + dH_diff_ub(eps_delta / 2**l, eps_rho / 2**l)
                    n_points_to_retain = Qs[l].bisect_left((dH_thresh, ))    # retain if < dH_thresh
                    for _ in range(len(Qs[l]) - n_points_to_retain):
                        Qs[l].pop()
                    if n_points_to_retain == 0 and l == shallowest_level:
                        shallowest_level += 1
                        if verbose:
                            print(f'{shallowest_level=}/{m}')

            # If no child point is a better candidate to zoom in on...
            if (best_child_dH >= best_dH or level == m) and shallowest_level < m:
                # ...Find the level of best known grid point.
                candidate_dH = min(Qs[l][0] for l in range(shallowest_level, m) if Qs[l])
                level = max(l for l in range(shallowest_level, m) if Qs[l] and Qs[l][0] == candidate_dH)

            best_dH = min(best_dH, best_child_dH)

    return best_dH
