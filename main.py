# --------------------- Create a bell-shaped surface (Gaussian bump) ---------------------
from shapes import create_bell_shape
from visualization import visualize
from mls import MLSParams, MovingLeastSquares


X_noisy, X = create_bell_shape(N=1000, A=1.0, sigma=0.6, noise_z_scale=0.03, noise_xy_scale=0.01, random_seed=123)

mls_params = MLSParams(
    k=120,
    degree=2,
    h_multiplier=6.0,
    return_normals=True
)

mls = MovingLeastSquares(X_noisy, mls_params)

# --------------------- Apply MLS smoothing ---------------------
X_smooth, N_smooth = mls.run()

visualize(X_noisy, X_smooth, N_smooth, A=1.0, sigma=0.6)
