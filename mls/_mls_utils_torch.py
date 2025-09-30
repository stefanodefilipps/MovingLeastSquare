from typing import Tuple
import torch
from typing_extensions import Literal


Degree = Literal[1, 2]

# -------------------------- small helpers --------------------------

def _design_xy(x: torch.Tensor, y: torch.Tensor, degree: Degree) -> torch.Tensor:
    """Build design matrix Phi for z=f(x,y) in batch: (N,k) -> (N,k,m)."""
    ones = torch.ones_like(x)
    if degree == 1:
        return torch.stack([ones, x, y], dim=-1)                      # m=3
    elif degree == 2:
        return torch.stack([ones, x, y, x*x, x*y, y*y], dim=-1)       # m=6
    else:
        raise ValueError("degree must be 1 or 2")

def _eval_poly(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, degree: Degree) -> torch.Tensor:
    """Evaluate z=f(x,y) with coeffs a: (...,m) -> (...)."""
    if degree == 1:
        a0,a1,a2 = a.unbind(-1)
        return a0 + a1*x + a2*y
    else:
        a0,a1,a2,a3,a4,a5 = a.unbind(-1)
        return a0 + a1*x + a2*y + a3*x*x + a4*x*y + a5*y*y

def _grad_poly(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, degree: Degree) -> torch.Tensor:
    """Gradient [dz/dx, dz/dy] at (x,y): (...,m) -> (...,2)."""
    if degree == 1:
        _,a1,a2 = a.unbind(-1)
        return torch.stack([a1, a2], dim=-1)
    else:
        a0,a1,a2,a3,a4,a5 = a.unbind(-1)
        dzdx = a1 + 2*a3*x + a4*y
        dzdy = a2 + a4*x + 2*a5*y
        return torch.stack([dzdx, dzdy], dim=-1)

def _pairwise_knn_chunked(P: torch.Tensor, k: int, chunk: int = 65536) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact kNN (self-self) in chunks, works on CPU or GPU.
    Returns d2 (N,k) squared distances, idx (N,k) indices (excluding self).
    """
    N = P.shape[0]
    d2_all, idx_all = [], []
    for s in range(0, N, chunk):
        e = min(s+chunk, N)
        A = P[s:e]                                  # (M,3)
        a2 = (A*A).sum(dim=1, keepdim=True)         # (M,1)
        b2 = (P*P).sum(dim=1).unsqueeze(0)          # (1,N)
        ab = A @ P.t()                               # (M,N)
        D2 = torch.clamp(a2 + b2 - 2*ab, min=0)
        d2, idx = torch.topk(D2, k=k+1, dim=1, largest=False)
        d2_all.append(d2[:,1:])                     # drop self
        idx_all.append(idx[:,1:])
    return torch.cat(d2_all, 0), torch.cat(idx_all, 0)

def _pairwise_knn_chunked_2(P: torch.Tensor, A: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact kNN (self-self) in chunks, works on CPU or GPU.
    Returns d2 (N,k) squared distances, idx (N,k) indices (excluding self).
    """
    a2 = (A*A).sum(dim=1, keepdim=True)         # (M,1)
    b2 = (P*P).sum(dim=1).unsqueeze(0)          # (1,N)
    ab = A @ P.t()                               # (M,N)
    D2 = torch.clamp(a2 + b2 - 2*ab, min=0)
    d2, idx = torch.topk(D2, k=k+1, dim=1, largest=False)
    return d2,idx

def _compute_pca(Q: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA (mean, rotation) for a batch of point clouds.
    Q: (N,k,3) -> c: (N,1,3), T: (N,3,3) with T=[t1,t2,n] where n is the smallest-variance direction.
    """

    # 2) Local PCA per neighborhood (batched 3x3 SVD on covariance)
    c = Q.mean(dim=1, keepdim=True)                            # (N,1,3)
    X = Q - c                                                  # (N,k,3)
    C = X.transpose(1,2).matmul(X) / max(k-1, 1)               # (N,3,3)
    U, S, Vh = torch.linalg.svd(C)                             # U: (N,3,3), S desc
    n  = U[..., 2]                                             # smallest-variance dir
    t1 = U[..., 0]; t2 = U[..., 1]
    T = torch.stack([t1, t2, n], dim=-1)

    return c, T