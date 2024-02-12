import numpy as np
from scipy import spatial as sp

from points import Points


class PointCloud(object):

    def __init__(self, coords, kd_tree=None):
        '''
        :param coords: point coordinates, (n, n_dim)-array
        :param kd_tree: k-d tree induced by the non-transformed points (or None to be computed)
        '''
        self.orig_centroid = coords.mean(axis=0)
        self.points = Points(coords - self.orig_centroid)

        # Obtain Voronoi tesselation of the feasible region.
        self.kd_tree = kd_tree or sp.KDTree(self.points.coords)

    def transform(self, T):
        transformed_points = self.points.transform(T)

        return PointCloud(transformed_points.coords, kd_tree=self.kd_tree)

    def asymm_dH(self, other):
        '''
        Find one-sided Hausdorff distance to another point cloud

        :param other: another point cloud, PointCloud
        :return: distance
        '''
        # Compute distances to the nearest neighbors in other.
        distances_to_other, _ = other.kd_tree.query(self.points.coords, workers=-1)

        return np.max(distances_to_other)
