import numpy as np
from scipy import spatial as sp


class PointCloud(object):

    def __init__(self, coords, orig_centroid=None, kd_tree=None):
        '''
        :param coords: point coordinates, (n, k)-array
        :param kd_tree: K-d tree on normalized points, sp.KDTree
        :param orig_centroid: pre-normalized centroid, k-array
        '''
        self.coords = np.array(coords, dtype=float)
        if orig_centroid is None:
            # Shift the centroid to the origin.
            self.orig_centroid = self.coords.mean(axis=0)
            self.coords -= self.orig_centroid
        else:
            self.orig_centroid = orig_centroid
        # Build k-d tree on the normalized points if needed.
        self.kd_tree = kd_tree or sp.KDTree(self.coords)

    def transform(self, T):
        transformed_coords = T.apply(self.coords)
        transformed_self = PointCloud(
            transformed_coords, orig_centroid=self.orig_centroid, kd_tree=self.kd_tree)

        return transformed_self

    def asymm_dH(self, other):
        '''
        Find one-sided Hausdorff distance to another point cloud

        :param other: another point cloud, PointCloud
        :return: distance
        '''
        # Compute distances to the nearest neighbors in other.
        distances_to_other, _ = other.kd_tree.query(self.coords, workers=-1)

        return np.max(distances_to_other)
