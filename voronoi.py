import numpy as np
from itertools import permutations


class Voronoi(object):

    def __init__(self, points, bbox, vertices, ridge_vert_idxs, ridge_point_idxs):
        '''
        :param points: points inducing Voronoi tesselation, Points
        :param bbox: bounding box, (n_dim, 2)-array
        :param vertices: vertices of Voronoi cells, Points
        :param ridge_vert_idxs: vertices of cell ridges, list of (2)-lists
        :param ridge_point_idxs: points separated by cell ridges, list of (2)-lists
        '''
        self.bbox = bbox
        self.bbox_coords = np.array(self.bbox).T
        self.vertices = vertices
        self.ridge_vert_idxs = np.array(ridge_vert_idxs)
        self.ridge_point_idxs = ridge_point_idxs

        # Prepare data structures for quick calculation of cell extents.
        cell_vert_idxs_by_p_idx = {p_idx: set() for p_idx in range(points.n)}
        unb_v_p2_idxs_by_p1_idx = {p_idx: set() for p_idx in range(points.n)}
        for (v1_idx, v2_idx), p_idxs in zip(ridge_vert_idxs, ridge_point_idxs):
            assert v2_idx >= 0, '2nd ridge vertex is -1'
            for p1_idx, p2_idx in permutations(p_idxs):
                if v1_idx < 0:
                    unb_v_p2_idxs_by_p1_idx[p1_idx].add((v2_idx, p2_idx))
                else:
                    cell_vert_idxs_by_p_idx[p1_idx].add(v1_idx)
                cell_vert_idxs_by_p_idx[p1_idx].add(v2_idx)

        self.cell_vert_idxs = []
        self.cell_vert_1st_idxs = []
        self.unb_cell_idxs = []
        self.unb_vert_idxs = []
        self.unb_p1_idxs = [] # same as unb_cell_idxs but repeated
        self.unb_p2_idxs = []
        self.bbox_vert_1st_idxs = []
        cell_1st_idx = unb_1st_idx = 0
        for p_idx in range(points.n):
            # Update data structures for bounded cells.
            cell_vert_idxs = list(cell_vert_idxs_by_p_idx[p_idx])
            self.cell_vert_idxs.extend(cell_vert_idxs)
            self.cell_vert_1st_idxs.append(cell_1st_idx)
            cell_1st_idx += len(cell_vert_idxs)
            assert cell_vert_idxs, f'cell {p_idx} has no vertices'

            # Update data structures for unbounded cells if needed.
            unb_v_p2_idxs = list(unb_v_p2_idxs_by_p1_idx[p_idx])
            if unb_v_p2_idxs:
                self.unb_cell_idxs.append(p_idx)
                self.unb_p1_idxs.extend([p_idx] * len(unb_v_p2_idxs))
                v_idxs, p2_idxs = zip(*unb_v_p2_idxs)
                self.unb_vert_idxs.extend(v_idxs)
                self.unb_p2_idxs.extend(p2_idxs)
                self.bbox_vert_1st_idxs.append(unb_1st_idx)
                unb_1st_idx += len(unb_v_p2_idxs)

        self.cell_min_coords, self.cell_max_coords = self.calc_cell_extents(points)

    def calc_cell_extents(self, points):
        '''
        Find min and max coordinates in each dimension of each Voronoi cell
        :param points: points inducing Voronoi tesselation, Points
        :return: (n, n_dim)-array of min coords, (n, n_dim)-array of max coords
        '''
        # Calculate cell extents based on bounded ridges.
        cell_min_coords = np.full((points.n, points.n_dim), np.inf)
        cell_max_coords = np.full((points.n, points.n_dim), -np.inf)
        cell_vert_coords = self.vertices.coords[self.cell_vert_idxs]
        for dim in range(points.n_dim):
            cell_min_coords[:, dim] = np.minimum.reduceat(
                cell_vert_coords[:, dim], self.cell_vert_1st_idxs)
            cell_max_coords[:, dim] = np.maximum.reduceat(
                cell_vert_coords[:, dim], self.cell_vert_1st_idxs)

        # Update cell extents based on unbounded ridges.
        n_unb_cells = len(self.unb_cell_idxs)
        unb_cell_min_coords = np.full((n_unb_cells, points.n_dim), np.inf)
        unb_cell_max_coords = np.full((n_unb_cells, points.n_dim), -np.inf)
        unb_vert_coords = self.vertices.coords[self.unb_vert_idxs]
        unb_midp_coords = (points.coords[self.unb_p1_idxs] + points.coords[self.unb_p2_idxs]) / 2
        assert not np.isclose(unb_vert_coords, unb_midp_coords).all(axis=1).any(), \
            f'unbounded ridge vertex coincides with its (p1+p2)/2'
        directions = unb_midp_coords - unb_vert_coords
        # Find the times for unbounded ridges to reach each bounding plane.
        ts = np.concatenate((self.bbox_coords[:, None] - unb_vert_coords) / directions, axis=1)
        # Find the first bounding plane reached by each unbounded ridge.
        ts[ts < 0] = np.inf
        ts = np.min(ts, axis=1)
        # Find the bbox vertices corresponding to these intersections.
        bbox_vert_coords = unb_vert_coords + ts[:, None] * directions
        # Update unbounded cell extents based on their bbox vertices.
        for dim in range(points.n_dim):
            unb_cell_min_coords[:, dim] = np.minimum.reduceat(
                bbox_vert_coords[:, dim], self.bbox_vert_1st_idxs)
            cell_min_coords[self.unb_cell_idxs] = np.minimum(
                cell_min_coords[self.unb_cell_idxs], unb_cell_min_coords)
            unb_cell_max_coords[:, dim] = np.maximum.reduceat(
                bbox_vert_coords[:, dim], self.bbox_vert_1st_idxs)
            cell_max_coords[self.unb_cell_idxs] = np.maximum(
                cell_max_coords[self.unb_cell_idxs], unb_cell_max_coords)

        return cell_min_coords, cell_max_coords

    def transform(self, points, nontriv_refl=None, theta=None, delta=None):
        '''
        Isometrically transform Voronoi tesselation.

        :param points: transformed points inducing Voronoi tesselation, Points
        :param nontriv_refl:
        :param theta:
        :param delta:
        :return: transformed Voronoi
        '''
        new_vertices = self.vertices.transform(
            nontriv_refl=nontriv_refl, theta=theta, delta=delta)

        return Voronoi(
            points, self.bbox, new_vertices, self.ridge_vert_idxs, self.ridge_point_idxs)
