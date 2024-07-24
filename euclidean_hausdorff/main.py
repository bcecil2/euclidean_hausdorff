from euclidean_hausdorff.eucl_haus import upper_heuristic
from euclidean_hausdorff.transformation import Transformation
from euclidean_hausdorff.utils import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    box = np.array([[1., 1.],
                    [-1., 1.],
                    [-1., -1.],
                    [1, -1]])
    t = Transformation(np.array([1, 2]),[np.pi/4],False)
    transformed_box = t.apply(box)
    plot(box, transformed_box)

    dist, e_ub = upper_heuristic(box,transformed_box,target_err=0.0,max_no_improv=10,verbose=2)
    print(dist)
    # hauss_d, diam = haus_dist(box, shifted, diam=True)
    # d, O, p, t = grid_search(box, shifted, diam,10)
    # print(d)
    # print(O, p, t)
    # #plot(box, shifted)
    # plot(distort(box, O, p, t), shifted)
    #
    # ax = plt.axes(projection="3d")
    #
    # cube = np.array([[0., 0., 0.],
    #                  [1., 0., 0.],
    #                  [1., 1., 0],
    #                  [0., 1., 0.],
    #                  [1., 0, 1.],
    #                  [1., 1., 1.],
    #                  [0., 0., 1.],
    #                  [0., 1., 1.]])
    #
    # O = rot3d((1., 0., 0.), np.pi / 4)
    # r = -np.eye(3)
    # t = np.array([0.3, 0.5, -0.2])
    # shift = distort(cube, O, r, t)
    # ax.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2])
    # ax.scatter3D(shift[:, 0], shift[:, 1], shift[:, 2])
    #
    # ax = plt.axes(projection="3d")
    # d, O_approx, r_approx, t_approx = grid_search3d(cube, shift, 2)
    # print(d)
    # print(O, p, t)
    # print(O_approx, r_approx, t_approx)
    # final = distort(cube, O_approx, r_approx, t_approx)
    # ax.scatter3D(final[:, 0], final[:, 1], final[:, 2])
    # ax.scatter3D(shift[:, 0], shift[:, 1], shift[:, 2])
    # plt.show()
