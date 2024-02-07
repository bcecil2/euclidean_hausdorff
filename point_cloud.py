import numpy as np
from scipy import spatial as sp
from itertools import product

from points import Points
from voronoi import Voronoi


class PointCloud(object):

    def __init__(self, coords, bbox=None, vor=None):
        '''
        :param coords: point coordinates, (n, n_dim)-array
        :param bbox: bounding box coordinates, (n_dim, 2)-
        :param vor: Voronoi tesselation of the bounding box induced by the points, Voronoi
        '''
        self.orig_centroid = coords.mean(axis=0)
        self.points = Points(coords - self.orig_centroid)

        # Obtain Voronoi tesselation of the feasible region.
        if vor is None:
            sp_vor = sp.Voronoi(self.points.coords)
            self.vor = Voronoi(self.points, bbox, Points(sp_vor.vertices),
                               sp_vor.ridge_vertices, sp_vor.ridge_points)
        else:
            self.vor = vor

        # Sort points by the coordinate extremes of their Voronoi cells.
        min_sorting_idxs = np.argsort(self.vor.cell_min_coords, axis=0).T
        max_sorting_idxs = np.argsort(self.vor.cell_max_coords, axis=0).T
        mins = np.take_along_axis(self.vor.cell_min_coords.T, min_sorting_idxs, axis=1)
        maxs = np.take_along_axis(self.vor.cell_max_coords.T, max_sorting_idxs, axis=1)

        # Initialize grid-based data structures for O(log n) point location.
        n, n_dim = self.points.n, self.points.n_dim
        approx_n_points_per_box_side = n ** (1 / n_dim)
        grid_resolution = round(np.ceil(np.log2(approx_n_points_per_box_side)))
        parent_gridcell = (0,) * n_dim
        mins_by_gridcell = {parent_gridcell: mins}
        maxs_by_gridcell = {parent_gridcell: maxs}
        min_sorting_idxs_by_gridcell = {parent_gridcell: min_sorting_idxs}
        max_sorting_idxs_by_gridcell = {parent_gridcell: max_sorting_idxs}
        self.point_idxs_by_gridcell = {parent_gridcell: np.arange(n)}

        # Fill in the data structures on the dyadic grid at incremental resolution for efficiency.
        for k in range(1, grid_resolution):
            new_point_idxs_by_gridcell = dict()
            new_mins_by_gridcell = dict()
            new_maxs_by_gridcell = dict()
            new_min_sorting_idxs_by_gridcell = dict()
            new_max_sorting_idxs_by_gridcell = dict()
            self.gridcell_sides = np.diff(self.vor.bbox).flatten() / 2**k
            # For each grid cell, find points whose Voronoi cells overlap it by
            # refining those for the parent grid cell.
            for gridcell in product(range(2**k), repeat=n_dim):
                gridcell_arr = np.array(gridcell)
                gridcell_start = self.vor.bbox_coords[0] + self.gridcell_sides * gridcell_arr
                gridcell_end = gridcell_start + self.gridcell_sides

                # Identify points whose Voronoi cells overlap the grid cell.
                parent_gridcell = tuple(gridcell_arr // 2)
                mins = mins_by_gridcell[parent_gridcell]
                min_sorting_idxs = min_sorting_idxs_by_gridcell[parent_gridcell]
                maxs = maxs_by_gridcell[parent_gridcell]
                max_sorting_idxs = max_sorting_idxs_by_gridcell[parent_gridcell]
                point_idxs = self.point_idxs_by_gridcell[parent_gridcell]
                for dim in range(n_dim):
                    # Discard points whose Voronoi cell's min coordinate is after the grid cell.
                    is_min_retained = np.isin(min_sorting_idxs[dim], point_idxs)
                    idx_end = np.searchsorted(mins[dim][is_min_retained],
                                              gridcell_end[dim], side='right')
                    point_idxs = min_sorting_idxs[dim][is_min_retained][:idx_end]

                    # Discard points whose Voronoi cell's max coordinate is before the grid cell.
                    is_max_retained = np.isin(max_sorting_idxs[dim], point_idxs)
                    idx_start = np.searchsorted(maxs[dim][is_max_retained],
                                                gridcell_start[dim], side='left')
                    point_idxs = max_sorting_idxs[dim][is_max_retained][idx_start:]

                new_point_idxs_by_gridcell[gridcell] = point_idxs
                is_min_retained = np.isin(min_sorting_idxs, point_idxs)
                new_mins_by_gridcell[gridcell] = mins[is_min_retained].reshape(n_dim, -1)
                new_min_sorting_idxs_by_gridcell[gridcell] = min_sorting_idxs[is_min_retained].reshape(n_dim, -1)
                is_max_retained = np.isin(max_sorting_idxs, point_idxs)
                new_maxs_by_gridcell[gridcell] = maxs[is_max_retained].reshape(n_dim, -1)
                new_max_sorting_idxs_by_gridcell[gridcell] = max_sorting_idxs[is_max_retained].reshape(n_dim, -1)

            mins_by_gridcell = new_mins_by_gridcell
            maxs_by_gridcell = new_maxs_by_gridcell
            min_sorting_idxs_by_gridcell = new_min_sorting_idxs_by_gridcell
            max_sorting_idxs_by_gridcell = new_max_sorting_idxs_by_gridcell
            self.point_idxs_by_gridcell = new_point_idxs_by_gridcell

    def transform(self, nontriv_refl=None, theta=None, delta=None):
        new_points = self.points.transform(
            nontriv_refl=nontriv_refl, theta=theta, delta=delta)
        new_vor = self.vor.transform(
            new_points, nontriv_refl=nontriv_refl, theta=theta, delta=delta)

        return PointCloud(new_points.coords, vor=new_vor)

    def dH_directional(self, other):
        '''
        Find directed Hausdorff distance from a point cloud

        :param other: another point cloud, PointCloud
        :return: distance
        '''
        # Compute grid cell indices of the points in other.
        other_gridcells_arr = ((other.points.coords - self.vor.bbox_coords[0]) //
                               self.gridcell_sides).astype(int)
        other_gridcells = map(tuple, other_gridcells_arr.tolist())

        # Obtain and arrange indices of own candidate points corresponding to these grid cells.
        candidate_idxs_by_other_idx = list(map(self.point_idxs_by_gridcell.get, other_gridcells))
        ns_candidates = list(map(len, candidate_idxs_by_other_idx))
        candidate_idxs = np.concatenate(candidate_idxs_by_other_idx)
        other_idxs = np.repeat(np.arange(other.points.n), ns_candidates)

        # Compute distances from the points in other to their respective candidate points.
        deltas = other.points.coords[other_idxs] - self.points.coords[candidate_idxs]
        distances_to_candidates = np.sum(deltas**2, axis=1)**.5

        # Choose the nearest candidate for every point in other.
        other_dlm_idxs = np.insert(np.cumsum(ns_candidates)[:-1], 0, 0)
        distances = np.minimum.reduceat(distances_to_candidates, other_dlm_idxs)

        return np.max(distances)

    def dH(self, other):
        return max(self.dH_directional(other), other.dH_directional(self))
