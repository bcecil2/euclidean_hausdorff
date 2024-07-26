from euclidean_hausdorff.eucl_haus import upper_heuristic
from euclidean_hausdorff.transformation import Transformation
from euclidean_hausdorff.utils import *

if __name__ == '__main__':
    # examples go here
    A = np.random.randn(100,3)
    B = np.random.randn(200,3)
    upper_heuristic(A,B)

