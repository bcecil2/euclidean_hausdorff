# euclidean-hausdorff

Given the coordinates of 2- or 3-dimensional point clouds $A, B \subset \mathbb{R}^k$ (where $k \in \{2, 3\}), estimates their Euclidean–Hausdorff distance (which itself is a relaxation and an upper bound of the Gromov–Hausdorff distance)

$$d_\text{EH}(X, Y) = \inf_{T:E(k)} d_\text{H}(T(A), B),$$

where the infimum is taken over all $k$-dimensional Euclidean isometries and $\dH$ is the Hausdorff distance in $\RR^k$.

The distance is estimated from above by discretizing the compact feasible region (of the above minimization) into a search grid, whose vertices each represent a combination of some translation, rotation, and reflection.