import numpy as np
from scipy import spatial as sp
from itertools import permutations
from geometry import rot2d, refl2d, voronoi_bbox


class PointCloud(object):

    def __init__(self, points, bbox):
        self.points = points
        self.n, self.n_dim = points.shape
        self.bbox = bbox

        # Obtain Voronoi tesselation of the feasible region.
        self.cells = voronoi_bbox(self.points, self.bbox)

        # Sort Voronoi cells by their coordinate extremes.
        self.mins = dict()
        self.maxs = dict()
        self.min_idxs = dict()
        self.max_idxs = dict()
        for dim in range(self.n_dim):
            dim_mins = [np.min(cell[:, dim]) for cell in self.cells]
            self.mins[dim], self.min_idxs[dim] = zip(*sorted(zip(dim_mins, range(self.n))))

            dim_maxs = [np.max(cell[:, dim]) for cell in self.cells]
            self.maxs[dim], self.max_idxs[dim] = zip(*sorted(zip(dim_maxs, range(self.n))))

    def reflect(self, nontriv):
        assert self.n_dim == 2, 'reflection in 3D not implemented'

        return PointCloud(self.points @ refl2d(nontriv).T, self.bbox)

    def rotate(self, theta):
        assert self.n_dim == 2, 'rotation in 3D not implemented'

        return PointCloud(self.points @ rot2d(theta).T, self.bbox)

    def shift(self, delta):

        return PointCloud(self.points + delta, self.bbox)

    def transform(self, nontriv_refl=None, theta=None, delta=None):
        if nontriv_refl is None:
            nontriv_refl = False
        if theta is None:
            theta = 0
        if delta is None:
            delta = np.zeros(self.n_dim)

        return self.reflect(nontriv_refl).rotate(theta).shift(delta)

    def d(self, point):
        candidate_mask = np.full(self.n, True)
        for dim in range(self.n_dim):
            # Remove candidate cells whose min coordinate is after the point.
            idx_end = np.searchsorted(self.mins[dim], point[dim], side='right')
            candidates_to_remove = self.min_idxs[dim][idx_end:]

            # Remove candidate cells whose max coordinate is before the point.
            idx_start = np.searchsorted(self.maxs[dim], point[dim], side='left')
            candidates_to_remove += self.max_idxs[dim][:idx_start]

            candidate_mask[list(candidates_to_remove)] = False

        dists = sp.distance.cdist([point], self.points[candidate_mask])

        return np.min(dists)

    def dH(self, other):
        dH_to, dH_from = [max(A.d(p) for p in B.points)
                          for A, B in permutations([self, other])]

        return max(dH_to, dH_from)
