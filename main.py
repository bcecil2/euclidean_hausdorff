from utils import distort,plot
import numpy as np
from geometry import rot2d,rot3d
from loss import haus_dist
from optimizers import grid_search,grid_search3d
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    box = np.array([[1., 1.],
                    [-1., 1.],
                    [-1., -1.],
                    [1, -1]])
    print(box.shape)
    shifted = distort(box, O=rot2d(np.pi / 3), p=np.array([[1., 0.], [0., -1.]]), t=np.array([1, 2]))
    plot(box, shifted)

    hauss_d, diam = haus_dist(box, shifted, diam=True)
    d, O, p, t = grid_search(box, shifted, diam,10)
    print(d)
    print(O, p, t)
    #plot(box, shifted)
    plot(distort(box, O, p, t), shifted)

    ax = plt.axes(projection="3d")

    cube = np.array([[0., 0., 0.],
                     [1., 0., 0.],
                     [1., 1., 0],
                     [0., 1., 0.],
                     [1., 0, 1.],
                     [1., 1., 1.],
                     [0., 0., 1.],
                     [0., 1., 1.]])

    O = rot3d((1., 0., 0.), np.pi / 4)
    r = -np.eye(3)
    t = np.array([0.3, 0.5, -0.2])
    shift = distort(cube, O, r, t)
    ax.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2])
    ax.scatter3D(shift[:, 0], shift[:, 1], shift[:, 2])

    ax = plt.axes(projection="3d")
    d, O_approx, r_approx, t_approx = grid_search3d(cube, shift, 2)
    print(d)
    print(O, p, t)
    print(O_approx, r_approx, t_approx)
    final = distort(cube, O_approx, r_approx, t_approx)
    ax.scatter3D(final[:, 0], final[:, 1], final[:, 2])
    ax.scatter3D(shift[:, 0], shift[:, 1], shift[:, 2])
    plt.show()
