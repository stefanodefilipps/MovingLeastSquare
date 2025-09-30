from typing import Tuple
import numpy as np


def create_bell_shape(N=1000, A=1.0, sigma=0.6, noise_z_scale=0.03, noise_xy_scale=0.01, random_seed=123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a bell-shaped surface (Gaussian bump) with noise.

    Parameters:
    - N: Number of points to sample.
    - A: Amplitude of the Gaussian bump.
    - sigma: Standard deviation of the Gaussian (controls width).
    - noise_z_scale: Standard deviation of the noise added to the z-values.
    - noise_xy_scale: Standard deviation of the noise added to the x and y-values.
    - random_seed: Seed for the random number generator.

    Returns:
    - X_noisy: Noisy 3D points sampled from the bell shape.
    - X_true: True surface points (same as X_clean in this case).
    """

    # --------------------- Create a bell-shaped surface (Gaussian bump) ---------------------
    # Sample a disk in xy
    rng = np.random.default_rng(random_seed)
    r = sigma * np.sqrt(rng.uniform(0, 1, N)) * 1.6  # concentrate more points near center but still cover area
    theta = rng.uniform(0, 2*np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z_true = A * np.exp(-(x**2 + y**2) / (2*sigma**2))

    # Build 3D points and add noise (both vertical and small lateral)
    X = np.column_stack([x, y, z_true])
    noise_z = rng.normal(scale=noise_z_scale, size=N)                 # vertical noise
    noise_xy = rng.normal(scale=noise_xy_scale, size=(N,2))            # lateral jitter
    X_noisy = np.column_stack([x, y, z_true]) + np.column_stack([noise_xy, noise_z])

    return X_noisy, X  # noisy points, true surface points