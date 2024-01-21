import numpy as np
from scipy import spatial as sp

from points import Points
from voronoi import Voronoi


class PointCloud(object):

    def __init__(self, coords, bbox=None, vor=None):
        '''
        :param coords: point coordinates, (n, n_dim)-array
        :param bbox: bounding box coordinates, (n_dim, 2)-
        :param vor: Voronoi tesselation of the bounding box induced by the points, Voronoi
        '''
        self.points = Points(coords)

        # Obtain Voronoi tesselation of the feasible region.
        if vor is None:
            sp_vor = sp.Voronoi(coords)
            self.vor = Voronoi(self.points, bbox, Points(sp_vor.vertices),
                               sp_vor.ridge_vertices, sp_vor.ridge_points)
        else:
            self.vor = vor

        # Sort points by the coordinate extremes of their Voronoi cells.
        min_sorting_idxs = np.argsort(self.vor.cell_min_coords, axis=0)
        max_sorting_idxs = np.argsort(self.vor.cell_max_coords, axis=0)
        self.mins = np.take_along_axis(self.vor.cell_min_coords, min_sorting_idxs, axis=0).T
        self.maxs = np.take_along_axis(self.vor.cell_max_coords, max_sorting_idxs, axis=0).T
        self.min_idxs = min_sorting_idxs.T
        self.max_idxs = max_sorting_idxs.T

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
        candidate_mask = np.full((other.points.n, self.points.n), True)
        candidate_idxs = np.arange(self.points.n)
        # For each dimension...
        for dim_mins, dim_min_idxs, dim_maxs, dim_max_idxs, dim_coords in (
                zip(self.mins, self.min_idxs, self.maxs, self.max_idxs, other.points.coords.T)):
            # Discard points whose Voronoi cell's min coordinate is after the other's point.
            idx_ends = np.searchsorted(dim_mins, dim_coords, side='right')
            other_idxs, sorted_cell_idxs = np.where(candidate_idxs >= idx_ends[:, None])
            candidate_mask[other_idxs, dim_min_idxs[sorted_cell_idxs]] = False

            # Discard points whose Voronoi cell's max coordinate is before the other's point.
            idx_starts = np.searchsorted(dim_maxs, dim_coords, side='left')
            other_idxs, sorted_cell_idxs = np.where(candidate_idxs < idx_starts[:, None])
            candidate_mask[other_idxs, dim_max_idxs[sorted_cell_idxs]] = False

        other_idxs, candidate_idxs = np.where(candidate_mask)
        deltas = other.points.coords[other_idxs] - self.points.coords[candidate_idxs]
        distances_to_candidates = np.sum(deltas**2, axis=1)**.5
        ns_candidates = candidate_mask.sum(axis=1)
        other_dlm_idxs = np.insert(np.cumsum(ns_candidates)[:-1], 0, 0)
        distances = np.minimum.reduceat(distances_to_candidates, other_dlm_idxs)

        return np.max(distances)

    def dH(self, other):
        return max(self.dH_directional(other), other.dH_directional(self))
