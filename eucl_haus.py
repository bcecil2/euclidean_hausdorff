import numpy as np
from scipy import optimize
from itertools import product

from point_cloud import PointCloud
from transformation import Transformation


def calc_optimal_covering_radii(alpha, r):
    """
    Calculate covering radii for the translation and rotation spaces guaranteeing
    additive approximation error ≤αlpha and minimizing the product of their cardinalities.

    :param alpha: (upper bound of) additive approximation error, float
    :param r: maximum point norm, float
    :return: epsilon_delta, epsilon_rho
    """
    def obj_grad(eps_delta):
        return (np.arccos(1 - (alpha - eps_delta)**2 / (2 * r**2)) - 2 * eps_delta /
                np.sqrt(4 * r**2 - (alpha - eps_delta)**2))

    eps_delta, = optimize.fsolve(obj_grad, alpha/2)
    eps_rho = np.arccos(1 - (alpha - eps_delta)**2 / (2 * r**2))

    assert np.isclose(alpha, eps_delta + np.sqrt(2*(1 - np.cos(eps_rho)))*r)

    return eps_delta, eps_rho


def make_grid(center, cell_size, extent, ball_rad):
    """
    Compile a grid with cell size a on the intersection of cube [-l/2, l/2]^k + {c} and ball B(0, r).

    :param center: cube center c, k-array
    :param cell_size: cell side length a, float
    :param extent: cube side length l, float
    :param ball_rad: ball radius r, float
    :return: (?, k)-array of grid vertices, cell_size
    """
    # Reduce cell size without increasing the cell count.
    n_cells = int(np.ceil(extent / cell_size))
    cell_size = extent / n_cells

    # Calculate vertex positions separately in each dimension.
    vert_offsets = np.linspace(-extent/2, extent/2, n_cells)
    vert_positions = np.add.outer(center, vert_offsets)

    # Generate vertex coordinates.
    k = len(vert_positions)
    vertex_coords = np.reshape(np.meshgrid(*vert_positions), (k, -1)).T

    # Retain only the vertices covering the ball.
    lengths = np.linalg.norm(vertex_coords, axis=1)
    # TODO: exceed ball_rad by one "layer" of the grid to cover the boundary.
    vertex_coords = vertex_coords[lengths <= ball_rad]

    return vertex_coords, cell_size


def dH(A, B, T): # dH(T(A), B)
    return max(A.transform(T).asymm_dH(B),
               B.transform(T.invert()).asymm_dH(A))


def search(A, B, sigmas, rhos, deltas):
    """
    Search the transformation space provided.

    :param A:
    :param B:
    :param sigmas:
    :param rhos:
    :param deltas:
    :return:
    """
    min_haus_dist = np.inf
    for sigma, rho, delta in product(sigmas, rhos, deltas):
        T = Transformation(delta, rho, sigma)
        min_haus_dist = min(dH(A, B, T), min_haus_dist)

    return min_haus_dist


def approx_eucl_haus(A_coords, B_coords, alpha, proper_rigid=False):
    """
    Approximate the Euclidean–Hausdorff distance.

    :param A_coords: points of A, (?×k)-array
    :param B_coords: points of B, (?×k)-array
    :param alpha: (upper bound of) additive approximation error, float
    :param proper_rigid: whether to consider only proper rigid transformations, bool
    :return: approximate distance
    """
    A, B = PointCloud(A_coords), PointCloud(B_coords)
    normalized_coords = np.concatenate([A.coords, B.coords])

    _, k = normalized_coords.shape
    assert k in {2, 3}, 'only 2D and 3D spaces are supported'
    r = np.linalg.norm(normalized_coords, axis=1).max()

    eps_delta, eps_rho = calc_optimal_covering_radii(alpha, r)

    # Make translation grid.
    center = np.zeros(k)
    cell_size = 2 * eps_delta / np.sqrt(k)
    delta_grid, delta_cell_size = make_grid(center, cell_size, 2*r, r)

    # Make rotation grid.
    center = np.zeros(k * (k-1) // 2)
    cell_size = 2 * eps_rho / np.sqrt(k)
    rho_grid, rho_cell_size = make_grid(center, cell_size, 2*np.pi, np.pi)

    print(f'{r=:.2f}, |∆|={len(delta_grid)} (ε_δ={eps_delta:.2f}),'
          f'|P|={len(rho_grid)} (ε_ρ={eps_rho:.2f})')

    # Make reflection "grid".
    sigmas = [False] if proper_rigid else [False, True]

    # Minimize dH(T(A), B) over the grid product.
    dEH = search(A, B, sigmas, rho_grid, delta_grid)

    return dEH