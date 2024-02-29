from geometry import rot2d, refl2d, refl3d, rot3d


class Points(object):

    def __init__(self, coords):
        self.n, self.k = coords.shape
        self.coords = coords

    def reflect(self, reflection):
        assert self.k in [2,3], 'reflection is only implemented in 2D and 3D'
        if self.k == 2:
            return Points(self.coords @ refl2d(reflection).T)
        if self.k == 3:
            return Points(self.coords @ refl3d(reflection).T)

    def rotate(self, theta):
        assert self.k in [2,3], 'rotation is only implemented in 2D and 3D'
        if self.k == 2:
            return Points(self.coords @ rot2d(theta[0]).T)
        if self.k == 3:
            return Points(self.coords @ rot3d(theta[0],theta[1]).T)

    def shift(self, delta):
        return Points(self.coords + delta)

    def transform(self, T):
        return self.reflect(T.reflection).rotate(T.theta).shift(T.delta)
