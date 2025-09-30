from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ._mls_utils import Degree, NeighborhoodMode, _eval_poly_xy, _pca_frame, _poly_design_xy, _poly_grad_xy
from scipy.spatial import cKDTree

from mls.mls import MovingLeastSquares


@dataclass
class MLSParams:
    k: int = 30
    radius: Optional[float] = None
    degree: Degree = 2
    h_multiplier: float = 1.0
    return_normals: bool = True
    neighborhood_mode: NeighborhoodMode = "knn"  # or "radius"

class MovingLeastSquaresCPU(MovingLeastSquares):
    """
    Moving Least Squares (MLS) smoother for 3D point clouds.

    Usage:
        mls = MovingLeastSquares(points, MLSParams(k=40, degree=2, h_multiplier=1.2))
        proj, normals = mls.run()  # smooth all points
        # or
        proj_i, normal_i = mls.project_point(i)  # per-point projection

    You can also pass precomputed neighbors via set_neighbors().
    """

    def __init__(self, points: np.ndarray, params: MLSParams = MLSParams()):
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must be (N,3)")
        self.P = P
        self.N = len(P)
        self.params = params
        self.tree: Optional[cKDTree] = None
        self._local_scale: Optional[np.ndarray] = None
        self._knn_idx: Optional[np.ndarray] = None  # (N,k)
        self._knn_dist: Optional[np.ndarray] = None # (N,k) Euclidean or squared — your choice

    # --------- Neighbor search management ---------

    def build_index(self):
        """Build KD-tree if you will use internal neighbor queries."""
        self.tree = cKDTree(self.P)

    def set_neighbors(self, knn_idx: np.ndarray, knn_dist: Optional[np.ndarray] = None):
        """
        Provide precomputed neighbors to skip KD-tree queries (for speed).
        knn_idx: (N,k) indices
        knn_dist: (N,k) distances from P[i] to neighbors (optional; if None, computed on the fly)
        """
        if knn_idx.shape[0] != self.N:
            raise ValueError("knn_idx must have N rows")
        self._knn_idx = knn_idx
        self._knn_dist = knn_dist

    def _compute_local_scale(self, k0: Optional[int] = None):
        """Per-point local scale from small-k distances -> used for Gaussian bandwidth."""
        if self._local_scale is not None:
            return
        if self.tree is None:
            self.build_index()
        k = self.params.k
        k0 = k0 or min(max(8, k // 2), max(8, k))
        dists_k0, _ = self.tree.query(self.P, k=min(k0 + 1, self.N))
        if dists_k0.ndim == 1:
            dists_k0 = dists_k0[None, :]
        self._local_scale = np.median(dists_k0[:, 1:], axis=1)  # ignore self

    def _compute_weights(self, h: float, dq: np.ndarray, i: int, p: np.ndarray, Q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Gaussian weights from distances dq and bandwidth h."""
        # Weights (Gaussian in 3D distance from query point)
        self._compute_local_scale()
        h = self.params.h_multiplier * max(self._local_scale[i], 1e-12)
        dq = self._neighbor_dists(i, p, Q, idx)
        w = np.exp(-(dq * dq) / (h * h + 1e-18))
        return w

    def _transform_to_local(self, c: np.ndarray, T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform points to local frame defined by centroid c and basis T."""
        # Local coords of neighbors
        Qloc = (points - c) @ T
        xq, yq, zq = Qloc[:, 0], Qloc[:, 1], Qloc[:, 2]

        return xq, yq, zq
    
    def _compute_weighted_least_squares(self, xq: np.ndarray, yq: np.ndarray, zq: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute weighted least squares polynomial fit."""
        Phi = _poly_design_xy(xq, yq, self.params.degree)
        A = np.sqrt(w)[:, None] * Phi
        b = np.sqrt(w) * zq
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs
    
    def _compute_normal(self, coeffs: np.ndarray, T: np.ndarray, x: float, y: float) -> np.ndarray:

        # Normal from gradient
        fx, fy = _poly_grad_xy(coeffs, x, y, self.params.degree)
        n_loc = np.array([-fx, -fy, 1.0], dtype=np.float64)
        n_loc /= (np.linalg.norm(n_loc) + 1e-18)
        n_world = T @ n_loc
        n_world /= (np.linalg.norm(n_world) + 1e-18)
        if np.dot(n_world, T[:, 2]) < 0:
            n_world = -n_world
        return n_world
    
    def _evaluate_point(self, coeffs: np.ndarray, p: np.ndarray, c: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Project P[i]
        ploc = (p - c) @ T
        x, y = float(ploc[0]), float(ploc[1])
        z = _eval_poly_xy(coeffs, x, y, self.params.degree)
        proj_loc = np.array([x, y, z], dtype=np.float64)
        proj = c + proj_loc @ T.T

        if not self.params.return_normals:
            return proj, None
        
        n_world = self._compute_normal(coeffs, T, x, y)

        return proj, n_world

    # --------- Core per-point MLS ---------

    def project_points(self, i: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project point P[i] onto locally fitted surface; optionally return normal.
        """
        p = self.P[i]
        # Neighborhood
        idx = self._get_neighbors_for_point(i, p)
        Q = self.P[idx]
        if len(Q) < max(6, self.params.degree * 3):
            # Not enough points – return original
            normal = np.array([0, 0, 1], dtype=np.float64) if self.params.return_normals else None
            return p.copy(), normal

        # Local frame via PCA
        c, T = _pca_frame(Q)

        # Local coords of neighbors
        xq, yq, zq = self._transform_to_local(c, T, Q)

        # Weights (Gaussian in 3D distance from query point)
        w = self._compute_weights(self.params.h_multiplier, None, i, p, Q, idx)

        # Weighted least squares for z = f(x,y)
        coeffs = self._compute_weighted_least_squares(xq, yq, zq, w)

        # Project P[i]
        proj, n_world = self._evaluate_point(coeffs, p, c, T)

        return proj, n_world
    
    def run(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run MLS on all points in self.P and return (proj, normals or None).
        """
        proj = np.empty_like(self.P)
        normals = np.empty_like(self.P) if self.params.return_normals else None

        # Build index if needed, and local scales
        if self._knn_idx is None and self.tree is None:
            self.build_index()
        self._compute_local_scale()

        for i in range(self.N):
            pi, ni = self.project_points(i)
            proj[i] = pi
            if normals is not None:
                normals[i] = ni
        return proj, normals
    
    # --------- Internals: neighbors & distances ---------

    def _get_neighbors_for_point(self, i: int, p: np.ndarray) -> Sequence[int]:
        # Precomputed kNN?
        if self._knn_idx is not None:
            return self._knn_idx[i]

        # Else query KD-tree
        if self.tree is None:
            self.build_index()

        if self.params.neighborhood_mode == "radius" and self.params.radius is not None:
            idx = self.tree.query_ball_point(p, r=self.params.radius)
            if len(idx) < max(6, self.params.degree * 3):
                _, idx = self.tree.query(p, k=min(self.params.k, self.N))
                if np.isscalar(idx):
                    idx = [int(idx)]
            return idx
        else:
            _, idx = self.tree.query(p, k=min(self.params.k, self.N))
            if np.isscalar(idx):
                idx = [int(idx)]
            return idx

    def _neighbor_dists(self, i: int, p: np.ndarray, Q: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        # If we have precomputed distances for kNN, use them
        if self._knn_dist is not None:
            return self._knn_dist[i]
        # Otherwise compute Euclidean distances on the fly
        return np.linalg.norm(Q - p, axis=1)