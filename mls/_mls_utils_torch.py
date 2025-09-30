"""
Low-level PyTorch utilities for Moving Least Squares (MLS).

This module provides:
- Polynomial design, evaluation, and gradient functions in batch mode.
- Chunked exact kNN search (CPU or GPU) using pairwise distances.
- Batched PCA for local frames in point cloud neighborhoods.

These functions are used internally by GPU-based MLS implementations.
"""

from typing import Tuple
import torch
from typing_extensions import Literal

# Allowed polynomial degrees
Degree = Literal[1, 2]


# -------------------------- Polynomial helpers --------------------------

def _design_xy(x: torch.Tensor, y: torch.Tensor, degree: Degree) -> torch.Tensor:
    """
    Build the polynomial design matrix for MLS surface fitting.

    Parameters
    ----------
    x, y : torch.Tensor, shape (N, k)
        Local coordinates of neighbors in tangent plane.
    degree : Literal[1, 2]
        Polynomial degree (1 = linear, 2 = quadratic).

    Returns
    -------
    Phi : torch.Tensor, shape (N, k, m)
        Design matrix:
        - For degree=1: [1, x, y] (m=3)
        - For degree=2: [1, x, y, x^2, xy, y^2] (m=6)
    """
    ones = torch.ones_like(x)
    if degree == 1:
        return torch.stack([ones, x, y], dim=-1)                  # (N,k,3)
    elif degree == 2:
        return torch.stack([ones, x, y, x*x, x*y, y*y], dim=-1)   # (N,k,6)
    else:
        raise ValueError("degree must be 1 or 2")


def _eval_poly(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, degree: Degree) -> torch.Tensor:
    """
    Evaluate polynomial surface z = f(x,y).

    Parameters
    ----------
    a : torch.Tensor, shape (..., m)
        Polynomial coefficients per sample.
    x, y : torch.Tensor, shape (...)
        Local coordinates where surface is evaluated.
    degree : Literal[1, 2]
        Polynomial degree.

    Returns
    -------
    z : torch.Tensor, shape (...)
        Evaluated z values.
    """
    if degree == 1:
        a0, a1, a2 = a.unbind(-1)
        return a0 + a1 * x + a2 * y
    else:
        a0, a1, a2, a3, a4, a5 = a.unbind(-1)
        return a0 + a1 * x + a2 * y + a3 * x * x + a4 * x * y + a5 * y * y


def _grad_poly(a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, degree: Degree) -> torch.Tensor:
    """
    Compute gradient [dz/dx, dz/dy] of polynomial z = f(x,y).

    Parameters
    ----------
    a : torch.Tensor, shape (..., m)
        Polynomial coefficients.
    x, y : torch.Tensor, shape (...)
        Local coordinates where gradient is computed.
    degree : Literal[1, 2]
        Polynomial degree.

    Returns
    -------
    grad : torch.Tensor, shape (..., 2)
        Gradient [dz/dx, dz/dy].
    """
    if degree == 1:
        _, a1, a2 = a.unbind(-1)
        return torch.stack([a1, a2], dim=-1)
    else:
        a0, a1, a2, a3, a4, a5 = a.unbind(-1)
        dzdx = a1 + 2 * a3 * x + a4 * y
        dzdy = a2 + a4 * x + 2 * a5 * y
        return torch.stack([dzdx, dzdy], dim=-1)


# -------------------------- Neighbor search --------------------------

def _pairwise_knn_chunked(P: torch.Tensor, k: int, chunk: int = 65536) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact k-nearest neighbors (self-to-self) using pairwise distances, in chunks.

    Parameters
    ----------
    P : torch.Tensor, shape (N, 3)
        Input point cloud.
    k : int
        Number of neighbors (excluding self).
    chunk : int
        Chunk size for processing to save memory.

    Returns
    -------
    d2 : torch.Tensor, shape (N, k)
        Squared distances to k neighbors.
    idx : torch.Tensor, shape (N, k)
        Indices of k neighbors.
    """
    N = P.shape[0]
    d2_all, idx_all = [], []

    # Process in chunks to avoid OOM
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        A = P[s:e]                                 # (M,3)
        a2 = (A * A).sum(dim=1, keepdim=True)      # (M,1)
        b2 = (P * P).sum(dim=1).unsqueeze(0)       # (1,N)
        ab = A @ P.t()                             # (M,N)
        D2 = torch.clamp(a2 + b2 - 2 * ab, min=0)  # squared distances

        # Take k+1 because the closest neighbor is the point itself
        d2, idx = torch.topk(D2, k=k + 1, dim=1, largest=False)
        d2_all.append(d2[:, 1:])                   # drop self
        idx_all.append(idx[:, 1:])

    return torch.cat(d2_all, 0), torch.cat(idx_all, 0)


def _pairwise_knn_chunked_2(P: torch.Tensor, A: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact k-nearest neighbors between two sets of points (A vs P).

    Parameters
    ----------
    P : torch.Tensor, shape (N, 3)
        Reference point cloud.
    A : torch.Tensor, shape (M, 3)
        Query point cloud.
    k : int
        Number of neighbors to return.

    Returns
    -------
    d2 : torch.Tensor, shape (M, k+1)
        Squared distances to neighbors (includes self if A==P).
    idx : torch.Tensor, shape (M, k+1)
        Neighbor indices into P.
    """
    a2 = (A * A).sum(dim=1, keepdim=True)         # (M,1)
    b2 = (P * P).sum(dim=1).unsqueeze(0)          # (1,N)
    ab = A @ P.t()                                # (M,N)
    D2 = torch.clamp(a2 + b2 - 2 * ab, min=0)
    d2, idx = torch.topk(D2, k=k + 1, dim=1, largest=False)
    return d2, idx


# -------------------------- PCA --------------------------

def _compute_pca(Q: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute local PCA frames for a batch of neighborhoods.

    Parameters
    ----------
    Q : torch.Tensor, shape (N, k, 3)
        Neighborhoods (one per point).
    k : int
        Number of neighbors in each neighborhood.

    Returns
    -------
    c : torch.Tensor, shape (N, 1, 3)
        Centroid of each neighborhood.
    T : torch.Tensor, shape (N, 3, 3)
        Local basis for each neighborhood with columns [t1, t2, n],
        where n is the smallest-variance direction (normal).
    """
    # Centroid
    c = Q.mean(dim=1, keepdim=True)                      # (N,1,3)
    # Centered neighbors
    X = Q - c                                            # (N,k,3)
    # Covariance matrices
    C = X.transpose(1, 2).matmul(X) / max(k - 1, 1)      # (N,3,3)
    # SVD of covariance
    U, S, Vh = torch.linalg.svd(C)                       # U: (N,3,3)
    # Eigenvectors (already sorted by variance in SVD)
    n = U[..., 2]                                        # smallest-variance dir
    t1 = U[..., 0]
    t2 = U[..., 1]
    T = torch.stack([t1, t2, n], dim=-1)
    return c, T
