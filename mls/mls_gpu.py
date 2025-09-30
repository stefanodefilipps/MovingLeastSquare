from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from mls._mls_utils_torch import Degree, _compute_pca, _design_xy, _eval_poly, _grad_poly, _pairwise_knn_chunked, _pairwise_knn_chunked_2
from mls.mls import MovingLeastSquares
import time
from scipy.spatial import cKDTree

# -------------------------- parameters --------------------------

@dataclass
class MLSParamsGPU:
    k: int = 30
    degree: Degree = 2
    h_multiplier: float = 1.0
    return_normals: bool = True
    # If you pass FAISS neighbors, we’ll use them (recommended for big N)
    idx: Optional[torch.Tensor] = None   # (N,k) int64 on same device as points
    d2: Optional[torch.Tensor] = None    # (N,k) float32 squared distances
    batch_size:int=10000  # for large point clouds, process in batches of this size

class MovingLeastSquaresGPU(MovingLeastSquares):

    """
    Batched Moving Least Squares for 3D point clouds in PyTorch (CPU/GPU).

    - points: (N,3) float32 tensor
    - params: MLSParams (k, degree, h_multiplier, optional precomputed neighbors)

    Methods:
      run() -> (proj, normals or None)
    """

    def __init__(self, points: torch.Tensor, params: MLSParamsGPU = MLSParamsGPU(), device: Optional[str] = None):
        if not torch.is_tensor(points):
            raise TypeError("points must be a torch.Tensor of shape (N,3)")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (N,3)")
        self.device = device or (points.device.type if points.is_cuda else "cpu")
        self.P = points.to(self.device, dtype=torch.float32)  # (N,3)
        self.N = self.P.shape[0]
        self.params = params
        self.dtype = torch.float32
        self.tree = cKDTree(self.P.cpu().numpy())

    def _transform_to_local(self, Q: torch.Tensor, c: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        X = Q - c

        # 3) Transform neighbors to local coords
        Qloc = X.matmul(T)                                         # (N,k,3)
        xq, yq, zq = Qloc.unbind(-1)                               # each (N,k)
        return xq, yq, zq
    
    def _compute_weights(self, d2: torch.Tensor) -> torch.Tensor:
        local_scale = torch.median(torch.sqrt(d2 + 1e-12), dim=1).values  # (N,)
        h = (self.params.h_multiplier * torch.clamp_min(local_scale, 1e-12)).unsqueeze(1)   # (N,1)
        w = torch.exp(-d2 / (h*h + 1e-18))     
        return w

    def _compute_weighted_least_square(self, xq: torch.Tensor, yq: torch.Tensor, zq: torch.Tensor, w: torch.Tensor):

        Phi = _design_xy(xq, yq, self.params.degree)                           # (N,k,m)
        m = Phi.shape[-1]
        W = torch.sqrt(w).unsqueeze(-1)                            # (N,k,1)
        A = W * Phi                                                # (N,k,m)
        b = (W.squeeze(-1) * zq)                                   # (N,k)

        AtA = A.transpose(1,2).matmul(A)                           # (N,m,m)
        Atb = A.transpose(1,2).matmul(b.unsqueeze(-1)).squeeze(-1) # (N,m)
        AtA = AtA + 1e-6 * torch.eye(m, device=self.device, dtype=self.dtype).expand(self.params.batch_size, m, m)
        L = torch.linalg.cholesky(AtA)                             # (N,m,m)
        coeffs = torch.cholesky_solve(Atb.unsqueeze(-1), L).squeeze(-1)    # (N,m)
        return coeffs
    
    def _compute_normals(self, coeffs: torch.Tensor, x: torch.Tensor, y: torch.Tensor, T: torch.Tensor):

        g = _grad_poly(coeffs, x, y, self.params.degree)                        # (N,2)
        fx, fy = g[...,0], g[...,1]
        n_loc = torch.stack([-fx, -fy, torch.ones_like(fx)], dim=-1)  # (N,3)
        n_loc = n_loc / torch.clamp_min(torch.linalg.norm(n_loc, dim=-1, keepdim=True), 1e-12)
        normals = n_loc.unsqueeze(1).matmul(T.transpose(1,2)).squeeze(1)   # (N,3)
        # orient similar to PCA normal:
        n = T[:,:,2]
        s = torch.sign((normals * n).sum(dim=-1, keepdim=True)).clamp(min=-1, max=1)
        normals = normals * s

        return normals

    def project_points(self, P: torch.Tensor, coeffs: torch.Tensor, c: torch.Tensor, T: torch.Tensor):

        ploc = (P - c.squeeze(1)).unsqueeze(1).matmul(T).squeeze(1) # (N,3)
        x, y = ploc[...,0], ploc[...,1]
        z = _eval_poly(coeffs, x, y, self.params.degree)                        # (N,)
        proj_loc = torch.stack([x, y, z], dim=-1)                   # (N,3)
        proj = c.squeeze(1) + proj_loc.unsqueeze(1).matmul(T.transpose(1,2)).squeeze(1)

        if not self.params.return_normals:
            return proj, None
        
        normals = self._compute_normals(coeffs, x, y, T)  # (N,3)
        return proj, normals
    
    @torch.no_grad()
    def run(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        #N, P = self.N, self.P
        device = self.P.device

        proj = torch.empty_like(self.P)
        normals = torch.empty_like(self.P)

        for s in range(0, self.N, self.params.batch_size):
            e = min(s + self.params.batch_size, self.N)

            P = self.P[s:e]  # (M,3)
            N = P.shape[0]

            #d2,idx = _pairwise_knn_chunked_2(self.P, P, k=k)  # exact, chunked

            # query neighbors
            d2, idx = self.tree.query(P.cpu().numpy(), k=self.params.k)

            # convert back to torch
            idx = torch.from_numpy(idx).to(device=device, dtype=torch.int32)
            d2 = torch.from_numpy(d2).to(device=device, dtype=torch.float32)

            # Gather neighbors: Q: (N,k,3)
            Q = self.P[idx]

            start = time.time()
            c, T = _compute_pca(Q,self.params.k)                       # (N,3,3)

            # 3) Transform neighbors to local coords
            xq, yq, zq = self._transform_to_local(Q, c, T)

            # 4) Gaussian weights using distance from query point
            w = self._compute_weights(d2)                              # (N,k)

            # 5) Weighted least squares fit z = f(x,y)
            coeffs = self._compute_weighted_least_square(xq, yq, zq, w)  # (N,m)

            # 6) Project original points
            ploc = (P - c.squeeze(1)).unsqueeze(1).matmul(T).squeeze(1) # (N,3)
            x, y = ploc[...,0], ploc[...,1]
            z = _eval_poly(coeffs, x, y, self.params.degree)                        # (N,)
            proj_loc = torch.stack([x, y, z], dim=-1)                   # (N,3)
            proj_ = c.squeeze(1) + proj_loc.unsqueeze(1).matmul(T.transpose(1,2)).squeeze(1)

            proj[s:e] = proj_

            if not self.params.return_normals:
                continue

            # 7) Normals from gradient
            normals_ = self._compute_normals(coeffs, x, y, T)  # (N,3)
            normals[s:e] = normals_
            end = time.time()
            print(f"MLS Time: {end - start:.4f} seconds")
        return proj, normals