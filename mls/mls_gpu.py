"""
Moving Least Squares (MLS) GPU implementation for smoothing and normal estimation
on 3D point clouds using PyTorch. Supports both k-NN and radius-based neighborhoods,
with batching to avoid GPU memory overflow.

Author: <your name>
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from mls._mls_utils import NeighborhoodMode
from mls._mls_utils_torch import (
    Degree,
    _compute_pca,
    _design_xy,
    _eval_poly,
    _grad_poly,
    _pairwise_knn_chunked,
    _pairwise_knn_chunked_2,
)
from mls.mls import MovingLeastSquares
import time
from scipy.spatial import cKDTree

# -------------------------- parameters --------------------------

@dataclass
class MLSParamsGPU:
    """
    Parameters controlling GPU-based Moving Least Squares.

    Attributes
    ----------
    k : int
        Number of neighbors for k-NN queries.
    degree : Degree
        Polynomial degree (1=linear, 2=quadratic).
    h_multiplier : float
        Bandwidth scaling factor for Gaussian weighting.
    return_normals : bool
        Whether to compute normals along with smoothed points.
    idx : Optional[torch.Tensor]
        Precomputed neighbor indices (N,k) on device.
    d2 : Optional[torch.Tensor]
        Precomputed squared distances (N,k) on device.
    batch_size : int
        Number of points to process per batch (to save GPU memory).
    neighborhood_mode : NeighborhoodMode
        Either "knn" or "radius".
    radius : Optional[float]
        Search radius for "radius" mode.
    """
    k: int = 30
    degree: Degree = 2
    h_multiplier: float = 1.0
    return_normals: bool = True
    idx: Optional[torch.Tensor] = None
    d2: Optional[torch.Tensor] = None
    batch_size: int = 10000
    neighborhood_mode: NeighborhoodMode = "knn"
    radius: Optional[float] = None


class MovingLeastSquaresGPU(MovingLeastSquares):
    """
    Batched Moving Least Squares for 3D point clouds in PyTorch (CPU/GPU).

    Parameters
    ----------
    points : np.ndarray
        Input point cloud of shape (N,3).
    params : MLSParamsGPU
        Parameters controlling MLS behavior.
    device : Optional[str]
        Device to run on ("cpu" or "cuda").

    Methods
    -------
    run() -> (proj, normals or None)
        Run MLS smoothing over all points.
    """

    def __init__(self, points: np.ndarray, params: MLSParamsGPU = MLSParamsGPU(), device: Optional[str] = None):
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a np.ndarray of shape (N,3)")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (N,3)")

        self.device = device or "cpu"
        # Store point cloud as torch tensor
        self.P = torch.from_numpy(points).to(self.device, dtype=torch.float32)  # (N,3)
        self.P_np = self.P.cpu().numpy()
        self.N = self.P.shape[0]
        self.params = params
        self.dtype = torch.float32
        # CPU KD-tree for neighbor search (used in both kNN and radius)
        self.tree = cKDTree(self.P.cpu().numpy())

    # ---------------- helper computations ----------------

    def _transform_to_local(self, Q: torch.Tensor, c: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform neighbor points Q into the local coordinate system.

        Returns
        -------
        (xq, yq, zq) : tuple of torch.Tensor
            Coordinates of neighbors in the local PCA frame.
        """
        X = Q - c
        Qloc = X.matmul(T)  # (N,k,3)
        xq, yq, zq = Qloc.unbind(-1)
        return xq, yq, zq
    
    def _compute_weights(self, d2: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian weights based on squared distances.

        Parameters
        ----------
        d2 : torch.Tensor, shape (N,k)
            Squared distances to neighbors.

        Returns
        -------
        w : torch.Tensor, shape (N,k)
            Gaussian weights for each neighbor.
        """
        local_scale = torch.median(torch.sqrt(d2 + 1e-12), dim=1).values  # (N,)
        h = (self.params.h_multiplier * torch.clamp_min(local_scale, 1e-12)).unsqueeze(1)
        w = torch.exp(-d2 / (h*h + 1e-18))
        return w

    def _compute_weighted_least_square(self, xq: torch.Tensor, yq: torch.Tensor, zq: torch.Tensor, w: torch.Tensor):
        """
        Fit polynomial z=f(x,y) with weighted least squares.

        Returns
        -------
        coeffs : torch.Tensor, shape (N,m)
            Polynomial coefficients.
        """
        Phi = _design_xy(xq, yq, self.params.degree)  # (N,k,m)
        m = Phi.shape[-1]
        W = torch.sqrt(w).unsqueeze(-1)
        A = W * Phi
        b = (W.squeeze(-1) * zq)

        # Normal equations
        AtA = A.transpose(1,2).matmul(A)           # (N,m,m)
        Atb = A.transpose(1,2).matmul(b.unsqueeze(-1)).squeeze(-1)  # (N,m)

        # Regularization to prevent singularities
        AtA = AtA + 1e-6 * torch.eye(m, device=self.device, dtype=self.dtype).expand(AtA.size()[0], m, m)

        # Solve with Cholesky
        L = torch.linalg.cholesky(AtA)
        coeffs = torch.cholesky_solve(Atb.unsqueeze(-1), L).squeeze(-1)
        return coeffs
    
    def _compute_normals(self, coeffs: torch.Tensor, x: torch.Tensor, y: torch.Tensor, T: torch.Tensor):
        """
        Compute normals from polynomial gradient and transform back to world coordinates.
        """
        g = _grad_poly(coeffs, x, y, self.params.degree)  # (N,2)
        fx, fy = g[...,0], g[...,1]

        # Gradient -> local normal
        n_loc = torch.stack([-fx, -fy, torch.ones_like(fx)], dim=-1)
        n_loc = n_loc / torch.clamp_min(torch.linalg.norm(n_loc, dim=-1, keepdim=True), 1e-12)

        # Transform back to world coords
        normals = n_loc.unsqueeze(1).matmul(T.transpose(1,2)).squeeze(1)

        # Orient consistently with PCA normal
        n = T[:,:,2]
        s = torch.sign((normals * n).sum(dim=-1, keepdim=True)).clamp(min=-1, max=1)
        normals = normals * s
        return normals

    def project_points(self, P: torch.Tensor, coeffs: torch.Tensor, c: torch.Tensor, T: torch.Tensor):
        """
        Project points onto fitted polynomial surface.

        Returns
        -------
        proj : torch.Tensor, shape (N,3)
            Projected points.
        normals : Optional[torch.Tensor], shape (N,3)
            Estimated normals if requested.
        """
        ploc = (P - c.squeeze(1)).unsqueeze(1).matmul(T).squeeze(1)  # (N,3)
        x, y = ploc[...,0], ploc[...,1]
        z = _eval_poly(coeffs, x, y, self.params.degree)
        proj_loc = torch.stack([x, y, z], dim=-1)

        # Transform back to world
        proj = c.squeeze(1) + proj_loc.unsqueeze(1).matmul(T.transpose(1,2)).squeeze(1)

        if not self.params.return_normals:
            return proj, None
        
        normals = self._compute_normals(coeffs, x, y, T)
        return proj, normals

    def _get_neighbors_for_point(self, p: torch.Tensor) -> Sequence[int]:
        """
        Get neighbor indices and distances for a query point.

        Supports both radius and kNN modes.
        """
        if self.params.neighborhood_mode == "radius" and self.params.radius is not None:
            idx = self.tree.query_ball_point(p, r=self.params.radius)
            Q = self.P_np[idx]
            d2 = np.linalg.norm(Q - p, axis=1)
            return d2, idx
        else:
            d2, idx = self.tree.query(p, k=min(self.params.k, self.N))
            return d2, idx
    
    # ---------------- main loop ----------------

    @torch.no_grad()
    def run(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run MLS smoothing over all points.

        Returns
        -------
        proj : np.ndarray, shape (N,3)
            Smoothed (projected) points.
        normals : Optional[np.ndarray], shape (N,3)
            Estimated normals if return_normals=True.
        """
        device = self.P.device
        proj = torch.empty_like(self.P)
        normals = torch.empty_like(self.P)

        for s in range(0, self.N, self.params.batch_size):
            e = min(s + self.params.batch_size, self.N)
            P = self.P[s:e]

            # Neighbor search on CPU
            d2, idx = self._get_neighbors_for_point(P.cpu().numpy())

            # Convert to torch
            idx = torch.from_numpy(idx).to(device=device, dtype=torch.int32)
            d2 = torch.from_numpy(d2).to(device=device, dtype=torch.float32)

            # Gather neighbors
            Q = self.P[idx]

            start = time.time()
            c, T = _compute_pca(Q, self.params.k)

            # Transform to local coordinates
            xq, yq, zq = self._transform_to_local(Q, c, T)

            # Weights
            w = self._compute_weights(d2)

            # Weighted LS fit
            coeffs = self._compute_weighted_least_square(xq, yq, zq, w)

            # Project original points
            ploc = (P - c.squeeze(1)).unsqueeze(1).matmul(T).squeeze(1)
            x, y = ploc[...,0], ploc[...,1]
            z = _eval_poly(coeffs, x, y, self.params.degree)
            proj_loc = torch.stack([x, y, z], dim=-1)
            proj_ = c.squeeze(1) + proj_loc.unsqueeze(1).matmul(T.transpose(1,2)).squeeze(1)
            proj[s:e] = proj_

            if not self.params.return_normals:
                continue

            # Normals
            normals_ = self._compute_normals(coeffs, x, y, T)
            normals[s:e] = normals_
            end = time.time()
            print(f"MLS Time (batch {s}:{e}): {end - start:.4f} seconds")

        return proj.cpu().numpy(), normals.cpu().numpy()
