"""
CPU Moving Least Squares (MLS) smoother for 3D point clouds.

This module implements the classic MLS pipeline:
  1) For each query point, gather a local neighborhood (kNN or radius).
  2) Build a local frame via PCA (two tangents + normal).
  3) Fit a polynomial surface z = f(x, y) in the local frame using Gaussian weights.
  4) Project the query point onto the fitted patch and (optionally) compute its normal.

Public API:
  - MLSParams: configuration for MLS.
  - MovingLeastSquaresCPU: main class. Call `run()` to smooth the whole cloud
    or `project_points(i)` for a single point.

Notes
-----
- Uses SciPy's cKDTree for neighbor searches (CPU).
- Precomputed neighbors can be injected via `set_neighbors(...)` for speed.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ._mls_utils import (
    Degree,
    NeighborhoodMode,
    _eval_poly_xy,
    _pca_frame,
    _poly_design_xy,
    _poly_grad_xy,
)
from mls.mls import MovingLeastSquares


@dataclass
class MLSParams:
    """
    Parameters controlling CPU-based Moving Least Squares.

    Attributes
    ----------
    k : int
        Number of neighbors for kNN queries. Ignored if using radius mode exclusively.
    radius : Optional[float]
        Neighborhood radius for 'radius' mode (ignored if neighborhood_mode='knn').
    degree : Degree
        Polynomial degree for local fit: 1 (plane) or 2 (quadratic).
    h_multiplier : float
        Bandwidth multiplier for Gaussian weights; larger -> smoother.
    return_normals : bool
        If True, compute normals from the fitted polynomial gradient.
    neighborhood_mode : NeighborhoodMode
        'knn' (default) or 'radius'.
    """
    k: int = 30
    radius: Optional[float] = None
    degree: Degree = 2
    h_multiplier: float = 1.0
    return_normals: bool = True
    neighborhood_mode: NeighborhoodMode = "knn"  # or "radius"


class MovingLeastSquaresCPU(MovingLeastSquares):
    """
    Moving Least Squares (MLS) smoother for 3D point clouds (NumPy + SciPy).

    Usage
    -----
    >>> mls = MovingLeastSquaresCPU(points, MLSParams(k=40, degree=2, h_multiplier=1.2))
    >>> proj, normals = mls.run()   # smooth entire cloud
    >>> # or, per-point:
    >>> proj_i, normal_i = mls.project_points(i)

    You can also pass precomputed neighbors via `set_neighbors()` to skip KD-tree queries.
    """

    def __init__(self, points: np.ndarray, params: MLSParams = MLSParams()):
        """
        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Input point cloud (float).
        params : MLSParams
            MLS configuration.
        """
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must be (N,3)")
        self.P = P
        self.N = len(P)
        self.params = params

        # Neighbor infra
        self.tree: Optional[cKDTree] = None
        self._local_scale: Optional[np.ndarray] = None  # per-point scale for bandwidth h
        self._knn_idx: Optional[np.ndarray] = None      # optional precomputed neighbors (N,k)
        self._knn_dist: Optional[np.ndarray] = None     # optional precomputed distances (N,k)

    # --------- Neighbor search management ---------

    def build_index(self) -> None:
        """Build a KD-tree for internal neighbor queries."""
        self.tree = cKDTree(self.P)

    def set_neighbors(self, knn_idx: np.ndarray, knn_dist: Optional[np.ndarray] = None) -> None:
        """
        Provide precomputed neighbors to skip KD-tree queries (for speed).

        Parameters
        ----------
        knn_idx : np.ndarray, shape (N, k)
            Integer neighbor indices for each point.
        knn_dist : Optional[np.ndarray], shape (N, k)
            Distances from P[i] to neighbors (optional). If None, distances are computed on the fly.
        """
        if knn_idx.shape[0] != self.N:
            raise ValueError("knn_idx must have N rows")
        self._knn_idx = knn_idx
        self._knn_dist = knn_dist

    def _compute_local_scale(self, k0: Optional[int] = None) -> None:
        """
        Compute a per-point local scale (median of small-k distances), used to set Gaussian bandwidth h.

        Parameters
        ----------
        k0 : Optional[int]
            Small k used to estimate local spacing. Defaults to ~k/2, min 8.
        """
        if self._local_scale is not None:
            return
        if self.tree is None:
            self.build_index()
        k = self.params.k
        k0 = k0 or min(max(8, k // 2), max(8, k))
        dists_k0, _ = self.tree.query(self.P, k=min(k0 + 1, self.N))  # includes self @ col 0
        if dists_k0.ndim == 1:
            dists_k0 = dists_k0[None, :]
        # median over neighbors, excluding self
        self._local_scale = np.median(dists_k0[:, 1:], axis=1)

    def _compute_weights(
        self,
        i: int,
        p: np.ndarray,
        Q: np.ndarray,
        idx: Sequence[int],
    ) -> np.ndarray:
        """
        Compute Gaussian weights for neighbors of point i.

        Parameters
        ----------
        i : int
            Index of the query point.
        p : np.ndarray, shape (3,)
            Query point coordinates.
        Q : np.ndarray, shape (k_i, 3)
            Neighbor coordinates (k_i may vary in radius mode).
        idx : Sequence[int]
            Neighbor indices (used only when precomputed distances are available).

        Returns
        -------
        w : np.ndarray, shape (k_i,)
            Gaussian weights for each neighbor.
        """
        self._compute_local_scale()
        # Bandwidth h is proportional to local spacing at point i
        h = self.params.h_multiplier * max(self._local_scale[i], 1e-12)

        # Distances: use precomputed if available, else compute now
        if self._knn_dist is not None and self._knn_idx is not None:
            # when idx comes from _knn_idx[i], we can index distances directly
            # NOTE: in radius mode precomputed kNN typically won't match; then we fall back below
            if isinstance(idx, np.ndarray) and idx.ndim == 1 and len(idx) == self._knn_dist.shape[1]:
                dq = self._knn_dist[i]
            else:
                dq = np.linalg.norm(Q - p, axis=1)
        else:
            dq = np.linalg.norm(Q - p, axis=1)

        # Gaussian weights based on 3D distance from p
        w = np.exp(-(dq * dq) / (h * h + 1e-18))
        return w

    def _transform_to_local(
        self,
        c: np.ndarray,
        T: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform points into the local PCA frame.

        Parameters
        ----------
        c : np.ndarray, shape (3,)
            Centroid of neighborhood.
        T : np.ndarray, shape (3,3)
            Local basis [t1, t2, n] as columns.
        points : np.ndarray, shape (k_i, 3)
            Neighbor coordinates.

        Returns
        -------
        xq, yq, zq : np.ndarray
            Local coordinates of neighbors in the PCA frame.
        """
        Qloc = (points - c) @ T
        xq, yq, zq = Qloc[:, 0], Qloc[:, 1], Qloc[:, 2]
        return xq, yq, zq

    def _compute_weighted_least_squares(
        self,
        xq: np.ndarray,
        yq: np.ndarray,
        zq: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        """
        Solve weighted least squares for the polynomial z = f(x, y).

        Returns
        -------
        coeffs : np.ndarray, shape (m,)
            Polynomial coefficients (m=3 for degree=1; m=6 for degree=2).
        """
        Phi = _poly_design_xy(xq, yq, self.params.degree)
        # Weighted system: A = sqrt(w) * Phi, b = sqrt(w) * z
        A = np.sqrt(w)[:, None] * Phi
        b = np.sqrt(w) * zq
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs

    def _compute_normal(
        self,
        coeffs: np.ndarray,
        T: np.ndarray,
        x: float,
        y: float,
    ) -> np.ndarray:
        """
        Compute world-frame normal from polynomial gradient at (x, y) in local coords.
        """
        fx, fy = _poly_grad_xy(coeffs, x, y, self.params.degree)
        n_loc = np.array([-fx, -fy, 1.0], dtype=np.float64)
        n_loc /= (np.linalg.norm(n_loc) + 1e-18)
        n_world = T @ n_loc
        n_world /= (np.linalg.norm(n_world) + 1e-18)
        # Consistent orientation with PCA normal to reduce flips
        if np.dot(n_world, T[:, 2]) < 0:
            n_world = -n_world
        return n_world

    def _evaluate_point(
        self,
        coeffs: np.ndarray,
        p: np.ndarray,
        c: np.ndarray,
        T: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Project the query point onto the fitted polynomial patch and (optionally) compute the normal.

        Returns
        -------
        proj : np.ndarray, shape (3,)
            Projected point in world coordinates.
        normal : Optional[np.ndarray], shape (3,)
            Estimated normal, or None if normals are disabled.
        """
        # Project P[i] by evaluating z=f(x,y) at the point's (x,y) in local frame
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
        Project point P[i] onto the locally fitted surface; optionally return its normal.

        Parameters
        ----------
        i : int
            Index of the point to project.

        Returns
        -------
        proj : np.ndarray, shape (3,)
            Projected (smoothed) point.
        normal : Optional[np.ndarray], shape (3,)
            Estimated normal, or None if normals are disabled.
        """
        p = self.P[i]

        # Neighborhood indices (kNN or radius)
        idx = self._get_neighbors_for_point(i, p)

        # Gather neighbors
        Q = self.P[idx]
        if len(Q) < max(6, self.params.degree * 3):
            # Not enough points to fit a stable model; fall back to original
            normal = np.array([0, 0, 1], dtype=np.float64) if self.params.return_normals else None
            return p.copy(), normal

        # Local frame via PCA
        c, T = _pca_frame(Q)

        # Local coords of neighbors
        xq, yq, zq = self._transform_to_local(c, T, Q)

        # Gaussian weights in 3D distance from query point
        w = self._compute_weights(i, p, Q, idx)

        # Weighted least squares fit for z = f(x,y)
        coeffs = self._compute_weighted_least_squares(xq, yq, zq, w)

        # Project and (optionally) compute normal
        proj, n_world = self._evaluate_point(coeffs, p, c, T)
        return proj, n_world

    def run(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run MLS for all points in the cloud.

        Returns
        -------
        proj : np.ndarray, shape (N,3)
            Smoothed point cloud.
        normals : Optional[np.ndarray], shape (N,3)
            Normals for each point, if enabled.
        """
        proj = np.empty_like(self.P)
        normals = np.empty_like(self.P) if self.params.return_normals else None

        # Build KD-tree if needed, and estimate local scales for bandwidth
        if self._knn_idx is None and self.tree is None:
            self.build_index()
        self._compute_local_scale()

        # Per-point MLS (CPU)
        for i in range(self.N):
            pi, ni = self.project_points(i)
            proj[i] = pi
            if normals is not None:
                normals[i] = ni

        return proj, normals

    # --------- Internals: neighbors & distances ---------

    def _get_neighbors_for_point(self, i: int, p: np.ndarray) -> np.ndarray:
        """
        Retrieve neighbor indices for point i, honoring the configured mode.

        Returns
        -------
        idx : np.ndarray, shape (k,) or variable-length in radius mode
            Integer indices of neighbors.
        """
        # Use precomputed kNN if available
        if self._knn_idx is not None:
            return self._knn_idx[i].astype(np.intp, copy=False)

        # Else use KD-tree
        if self.tree is None:
            self.build_index()

        if self.params.neighborhood_mode == "radius" and self.params.radius is not None:
            idx = self.tree.query_ball_point(p, r=self.params.radius)  # list of ints
            # Fallback to kNN if too sparse
            if len(idx) < max(6, self.params.degree * 3):
                _, idx = self.tree.query(p, k=min(self.params.k, self.N))
            # Normalize to 1D integer array
            idx = np.asarray(idx, dtype=np.intp)
            if np.isscalar(idx):
                idx = np.array([int(idx)], dtype=np.intp)
            return idx
        else:
            # kNN mode
            _, idx = self.tree.query(p, k=min(self.params.k, self.N))
            # Normalize shape and dtype
            if np.isscalar(idx):
                idx = np.array([int(idx)], dtype=np.intp)
            else:
                idx = np.asarray(idx, dtype=np.intp)
            return idx

    def _neighbor_dists(self, i: int, p: np.ndarray, Q: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        """
        Return neighbor distances for point i (utility kept for compatibility).

        This is used when a caller specifically needs distances outside of
        `_compute_weights`. In normal flow `_compute_weights` already covers it.
        """
        if self._knn_dist is not None:
            return self._knn_dist[i]
        return np.linalg.norm(Q - p, axis=1)
