import numpy as np
from euclidean_hausdorff.eucl_haus import upper_heuristic

if __name__ == '__main__':
    # Examples go here:
    A_coords = np.random.randn(100, 3)
    B_coords = np.random.randn(200, 3)
    upper_heuristic(A_coords, B_coords)

