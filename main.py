# --------------------- Create a bell-shaped surface (Gaussian bump) ---------------------
import torch
from shapes import create_bell_shape
from visualization import visualize
from mls.mls_cpu import MLSParams, MovingLeastSquaresCPU
from mls.mls_gpu import MLSParamsGPU, MovingLeastSquaresGPU
import time


X_noisy, X = create_bell_shape(N=100000, A=1.0, sigma=0.6, noise_z_scale=0.03, noise_xy_scale=0.05, random_seed=123)

mls_params = MLSParams(
    k=240,
    degree=2,
    h_multiplier=6.0,
    return_normals=True
)

mls = MovingLeastSquaresCPU(X_noisy, mls_params)

# --------------------- Apply MLS smoothing ---------------------
# start = time.time()
# X_smooth, N_smooth = mls.run()
# end = time.time()
# print(f"MLS CPU Time: {end - start:.4f} seconds")

# visualize(X_noisy, X_smooth, N_smooth, A=1.0, sigma=0.6)


mls_params_gpu = MLSParamsGPU(
    k=240,
    degree=2,
    h_multiplier=6.0,
    return_normals=True,
    batch_size=50000
)

device = "cuda" if torch.cuda.is_available() else "cpu"

X_noisy_torch = torch.from_numpy(X_noisy).to(device=device, dtype=torch.float32)

mls = MovingLeastSquaresGPU(X_noisy_torch, mls_params_gpu)

# --------------------- Apply MLS smoothing ---------------------
start = time.time()
X_smooth, N_smooth = mls.run()
end = time.time()
print(f"MLS GPU Time: {end - start:.4f} seconds")

X_smooth = X_smooth.cpu().numpy()
N_smooth = N_smooth.cpu().numpy()

visualize(X_noisy, X_smooth, N_smooth, A=1.0, sigma=0.6)
