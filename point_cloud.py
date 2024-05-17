import numpy as np
from scipy import spatial as sp


class PointCloud(object):

    def __init__(self, coords, build_kd_tree=True, distance_agg='max'):
        '''
        :param coords: point coordinates, (n, k)-array
        :param kd_tree: k-d tree to speed up Hausdorff distance (or None to be computed)
        '''
        self.orig_centroid = coords.mean(axis=0)
        self.coords = coords - self.orig_centroid
        if distance_agg == 'max':
            self.agg = np.max
        elif distance_agg == 'mean':
            self.agg = np.mean
        elif distance_agg == 'median':
            self.agg = np.median
        else:
            raise f'unknown aggregate function {distance_agg}'

        if build_kd_tree:
            # Build k-d tree on the non-transformed points.
            self.kd_tree = sp.KDTree(self.coords)

    def transform(self, T):
        transformed_coords = T.apply(self.coords)
        transformed_self = PointCloud(transformed_coords, build_kd_tree=False)
        transformed_self.orig_centroid = self.orig_centroid
        transformed_self.kd_tree = self.kd_tree

        return transformed_self

    def asymm_dH(self, other):
        '''
        Find one-sided Hausdorff distance to another point cloud

        :param other: another point cloud, PointCloud
        :return: distance
        '''
        # Compute distances to the nearest neighbors in other.
        distances_to_other, _ = other.kd_tree.query(self.coords, workers=-1)

        return self.agg(distances_to_other)
