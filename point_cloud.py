import numpy as np
from scipy import spatial as sp
from itertools import permutations

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
        self.min_idxs = min_sorting_idxs.T.tolist()
        self.max_idxs = max_sorting_idxs.T.tolist()

    def transform(self, nontriv_refl=None, theta=None, delta=None):
        new_points = self.points.transform(
            nontriv_refl=nontriv_refl, theta=theta, delta=delta)
        new_vor = self.vor.transform(
            new_points, nontriv_refl=nontriv_refl, theta=theta, delta=delta)

        return PointCloud(new_points.coords, vor=new_vor)

    def d(self, point):
        '''
        Find distance from a point

        :param point: point coordinates (n_dim)-array
        :return: distance
        '''
        candidate_mask = np.full(self.points.n, True)
        # For each dimension...
        for dim_mins, dim_min_idxs, dim_maxs, dim_max_idxs, coord in (
                zip(self.mins, self.min_idxs, self.maxs, self.max_idxs, point)):
            # Find points whose Voronoi cell's min coordinate is after the point.
            idx_end = np.searchsorted(dim_mins, coord, side='right')
            candidates_to_remove = dim_min_idxs[idx_end:]

            # Find points whose Voronoi cell's max coordinate is before the point.
            idx_start = np.searchsorted(dim_maxs, coord, side='left')
            candidates_to_remove += dim_max_idxs[:idx_start]

            # Mask the above points.
            candidate_mask[candidates_to_remove] = False

        dists = sp.distance.cdist([point], self.points.coords[candidate_mask])

        return np.min(dists)

    def dH(self, other):
        dH_to, dH_from = [max(A.d(p) for p in B.points.coords)
                          for A, B in permutations([self, other])]

        return max(dH_to, dH_from)
