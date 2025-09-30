from typing import Literal, Tuple

import numpy as np


NeighborhoodMode = Literal["knn", "radius"]
Degree = Literal[1, 2]

# ----------------------------- Low-level helpers -----------------------------

def _pca_frame(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA on (n,3) neighborhood -> (centroid c, basis U) where U=[t1,t2,n].
    'n' is the smallest-variance direction (local normal).
    """
    c = P.mean(axis=0)
    X = P - c
    C = (X.T @ X) / max(len(P) - 1, 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)  # ascending
    t1, t2, n = evecs[:, order[1]], evecs[:, order[2]], evecs[:, order[0]]
    U = np.stack([t1, t2, n], axis=1)
    if U[2, 2] < 0:  # optional flip for a more consistent "up"
        U[:, 2] *= -1.0
    return c, U

def _poly_design_xy(x: np.ndarray, y: np.ndarray, degree: Degree) -> np.ndarray:
    if degree == 1:
        return np.column_stack([np.ones_like(x), x, y])
    elif degree == 2:
        return np.column_stack([np.ones_like(x), x, y, x*x, x*y, y*y])
    else:
        raise ValueError("degree must be 1 or 2")

def _eval_poly_xy(coeffs: np.ndarray, x: float | np.ndarray, y: float | np.ndarray, degree: Degree):
    if degree == 1:
        a0, a1, a2 = coeffs
        return a0 + a1 * x + a2 * y
    else:
        a0, a1, a2, a3, a4, a5 = coeffs
        return a0 + a1 * x + a2 * y + a3 * x * x + a4 * x * y + a5 * y * y

def _poly_grad_xy(coeffs: np.ndarray, x: float, y: float, degree: Degree) -> np.ndarray:
    if degree == 1:
        _, a1, a2 = coeffs
        return np.array([a1, a2], dtype=np.float64)
    else:
        a0, a1, a2, a3, a4, a5 = coeffs
        dzdx = a1 + 2 * a3 * x + a4 * y
        dzdy = a2 + a4 * x + 2 * a5 * y
        return np.array([dzdx, dzdy], dtype=np.float64)