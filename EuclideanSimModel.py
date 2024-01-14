import numpy as np
from optimizers import grid_search

class EuclideanSim:

    def __init__(self, buckets, dist_params, aggregator=max, dist_computer=grid_search):
        """

        :param buckets: dict of (label,[point clouds]) where the list is the top k representatives of a given class
        :param dist_computer: method of computing the euclidean hausdorff distance
        :param dist_params: kwargs for computing the distance
        :param aggregator: aggregation function to be applied to each bucket, should take a list of floats and return a non negative float
        """
        self.buckets = buckets
        self.dist = dist_computer
        self.params = dist_params
        self.agg = aggregator

    def similarity(self, A):
        """
        Assigns a label to A based on its similarity with each of the buckets
        :param A:
        :return:
        """
        argMin = (0,1e9)
        for label,items in self.buckets.items():
            results = [self.dist(A, Y, **self.params) for Y in items]
            m = self.agg(results)
            if m < argMin[1]:
                argMin = (label,m)
        return argMin[0]