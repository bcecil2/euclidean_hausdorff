from geometry import rot2d, refl2d


class Points(object):

    def __init__(self, coords):
        self.n, self.k = coords.shape
        self.coords = coords

    def reflect(self, reflection):
        assert self.k == 2, 'reflection is only implemented in 2D'

        return Points(self.coords @ refl2d(reflection).T)

    def rotate(self, theta):
        assert self.k == 2, 'rotation is only implemented in 2D'

        return Points(self.coords @ rot2d(theta[0]).T)

    def shift(self, delta):
        return Points(self.coords + delta)

    def transform(self, T):
        return self.reflect(T.reflection).rotate(T.theta).shift(T.delta)
