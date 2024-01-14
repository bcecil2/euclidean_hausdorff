from sklearn.base import BaseEstimator
from loss import haus_dist
from sklearn.utils.estimator_checks import check_estimator



class EuclideanDistClassifier(BaseEstimator):
    def __init__(self, rotation=None, reflection=None, shift=None):
        self.rotation = rotation
        self.reflection = reflection
        self.shift = shift

    def fit(self, A, B):
        self.A = A
        return self
    def predict(self,X):
        return self.A@self.rotation@self.reflection + self.shift
