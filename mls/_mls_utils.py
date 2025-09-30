"""
Low-level utilities for Moving Least Squares (MLS).

This module provides:
- Type aliases for neighborhood search and polynomial degree.
- PCA-based local frame computation for point cloud neighborhoods.
- Polynomial design matrix, evaluation, and gradient functions
  used for local surface fitting.

These functions are intended as internal helpers for MLS implementations.
"""

from typing import Literal, Tuple
import numpy as np

# ----------------------------- Types -----------------------------

# Allowed neighborhood search modes
NeighborhoodMode = Literal["knn", "radius"]

# Allowed polynomial degrees for MLS local fits
Degree = Literal[1, 2]


# ----------------------------- Low-level helpers -----------------------------

def _pca_frame(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a local PCA frame for a neighborhood.

    Parameters
    ----------
    P : np.ndarray, shape (n, 3)
        Neighborhood points.

    Returns
    -------
    c : np.ndarray, shape (3,)
        Centroid of the neighborhood.
    U : np.ndarray, shape (3, 3)
        Local basis matrix with columns [t1, t2, n]:
        - t1, t2 : tangent directions (larger-variance directions)
        - n      : normal direction (smallest-variance direction)
    
    Notes
    -----
    - The normal is defined as the eigenvector with the smallest eigenvalue.
    - A sign flip is applied if the normal’s z-component is negative,
      to enforce a more consistent "upward" orientation.
    """
    # Centroid
    c = P.mean(axis=0)
    X = P - c
    # Covariance
    C = (X.T @ X) / max(len(P) - 1, 1)
    # Eigen decomposition
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)  # ascending: smallest -> largest
    t1, t2, n = evecs[:, order[1]], evecs[:, order[2]], evecs[:, order[0]]
    U = np.stack([t1, t2, n], axis=1)
    # Flip normal if it points downward
    if U[2, 2] < 0:
        U[:, 2] *= -1.0
    return c, U


def _poly_design_xy(x: np.ndarray, y: np.ndarray, degree: Degree) -> np.ndarray:
    """
    Construct the polynomial design matrix for MLS surface fitting.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
        Local coordinates of neighbors in tangent plane.
    degree : Literal[1, 2]
        Polynomial degree: 1 = linear, 2 = quadratic.

    Returns
    -------
    Phi : np.ndarray, shape (n, m)
        Design matrix:
        - For degree=1: [1, x, y]
        - For degree=2: [1, x, y, x^2, xy, y^2]
    """
    if degree == 1:
        return np.column_stack([np.ones_like(x), x, y])
    elif degree == 2:
        return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    else:
        raise ValueError("degree must be 1 or 2")


def _eval_poly_xy(coeffs: np.ndarray, x: float | np.ndarray, y: float | np.ndarray, degree: Degree):
    """
    Evaluate a polynomial surface z = f(x,y) at given (x,y).

    Parameters
    ----------
    coeffs : np.ndarray, shape (m,)
        Polynomial coefficients from least squares fit.
    x, y : float or np.ndarray
        Coordinates to evaluate.
    degree : Literal[1, 2]
        Polynomial degree.

    Returns
    -------
    z : float or np.ndarray
        Evaluated surface height(s).
    """
    if degree == 1:
        a0, a1, a2 = coeffs
        return a0 + a1 * x + a2 * y
    else:
        a0, a1, a2, a3, a4, a5 = coeffs
        return a0 + a1 * x + a2 * y + a3 * x * x + a4 * x * y + a5 * y * y


def _poly_grad_xy(coeffs: np.ndarray, x: float, y: float, degree: Degree) -> np.ndarray:
    """
    Compute the gradient of the polynomial surface z = f(x,y).

    Parameters
    ----------
    coeffs : np.ndarray, shape (m,)
        Polynomial coefficients.
    x, y : float
        Point in local coordinates where gradient is computed.
    degree : Literal[1, 2]
        Polynomial degree.

    Returns
    -------
    grad : np.ndarray, shape (2,)
        Gradient [dz/dx, dz/dy].
    """
    if degree == 1:
        _, a1, a2 = coeffs
        return np.array([a1, a2], dtype=np.float64)
    else:
        a0, a1, a2, a3, a4, a5 = coeffs
        dzdx = a1 + 2 * a3 * x + a4 * y
        dzdy = a2 + a4 * x + 2 * a5 * y
        return np.array([dzdx, dzdy], dtype=np.float64)
