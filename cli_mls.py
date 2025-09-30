# cli_mls.py
from __future__ import annotations
import argparse, time, sys, os
import numpy as np
import torch

from shapes import create_bell_shape
from visualization import visualize
from mls.mls_cpu import MLSParams as MLSParamsCPU, MovingLeastSquaresCPU
from mls.mls_gpu import MLSParamsGPU, MovingLeastSquaresGPU

# Optional Open3D (import only if requested)
def _maybe_import_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception as e:
        print(f"[warn] Open3D not available: {e}")
        return None

def parse_args():
    p = argparse.ArgumentParser(
        description="MLS smoothing for a synthetic bell-shaped point cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shape params
    p.add_argument("--N", type=int, default=10_000, help="number of points")
    p.add_argument("--A", type=float, default=1.0, help="bell height")
    p.add_argument("--sigma", type=float, default=0.6, help="bell width (Gaussian sigma)")
    p.add_argument("--noise-z", type=float, default=0.03, help="vertical noise stddev")
    p.add_argument("--noise-xy", type=float, default=0.05, help="lateral noise stddev")
    p.add_argument("--seed", type=int, default=123, help="random seed")

    # MLS params (common)
    p.add_argument("--k", type=int, default=120, help="neighbors for kNN")
    p.add_argument("--degree", type=int, choices=[1,2], default=2, help="polynomial degree")
    p.add_argument("--h", type=float, default=6.0, help="h_multiplier (Gaussian bandwidth scale)")
    p.add_argument("--no-normals", action="store_true", help="do not compute normals")

    # Device / backend
    back = p.add_argument_group("backend")
    back.add_argument("--device", choices=["auto","cpu","cuda"], default="auto",
                      help="where to run MLS")
    back.add_argument("--batch-size", type=int, default=50_000,
                      help="GPU chunk size (only used on CUDA backend)")
    back.add_argument("--mode", choices=["knn","radius"], default="knn",
                      help="neighborhood mode (GPU class supports both if implemented)")
    back.add_argument("--radius", type=float, default=None,
                      help="radius for 'radius' mode (ignored if mode=knn)")

    # I/O and viz
    io = p.add_argument_group("IO")
    io.add_argument("--save-ply", type=str, default=None,
                    help="path to save smoothed PLY (writes points and normals if present)")
    io.add_argument("--show-open3d", action="store_true", help="visualize with Open3D")
    io.add_argument("--show-noisy", action="store_true", help="also show the noisy cloud in Open3D")
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- create noisy bell shape
    X_noisy, X_true = create_bell_shape(
        N=args.N,
        A=args.A,
        sigma=args.sigma,
        noise_z_scale=args.noise_z,
        noise_xy_scale=args.noise_xy,
        random_seed=args.seed,
    )

    # --- choose backend (CPU/GPU)
    return_normals = not args.no_normals

    # Device resolution
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[warn] --device=cuda requested but CUDA not available; falling back to CPU.")
            device = "cpu"

    # Build MLS object
    if device == "cpu":
        mls_params = MLSParamsCPU(
            k=args.k,
            degree=args.degree,
            h_multiplier=args.h,
            return_normals=return_normals,
            radius=args.radius,
            neighborhood_mode=args.mode,   # ignored in your CPU class if not implemented
        )
        mls = MovingLeastSquaresCPU(X_noisy, mls_params)
    else:
        mls_params = MLSParamsGPU(
            k=args.k,
            degree=args.degree,
            h_multiplier=args.h,
            return_normals=return_normals,
            batch_size=args.batch_size,
            neighborhood_mode=args.mode,
            radius=args.radius,
        )
        mls = MovingLeastSquaresGPU(X_noisy, mls_params, device=device)

    # --- run MLS
    t0 = time.time()
    X_smooth, N_smooth = mls.run()
    t1 = time.time()
    print(f"[info] MLS on {device}: {t1 - t0:.3f} s | N={args.N} k={args.k} degree={args.degree} h={args.h}")

    # --- optional Open3D viz
    if args.show_open3d:
        o3d = _maybe_import_open3d()
        if o3d is None:
            print("[warn] Skipping Open3D visualization.")
        else:
            geoms = []
            if args.show_noisy:
                pcd_noise = o3d.geometry.PointCloud()
                pcd_noise.points = o3d.utility.Vector3dVector(X_noisy)
                pcd_noise.paint_uniform_color([1,0,0])  # red
                geoms.append(pcd_noise)

            pcd_smooth = o3d.geometry.PointCloud()
            pcd_smooth.points = o3d.utility.Vector3dVector(X_smooth)
            if return_normals and N_smooth is not None:
                pcd_smooth.normals = o3d.utility.Vector3dVector(N_smooth)
            pcd_smooth.paint_uniform_color([0,1,0])  # green
            geoms.append(pcd_smooth)

            o3d.visualization.draw_geometries(geoms, point_show_normal=return_normals)

if __name__ == "__main__":
    main()
