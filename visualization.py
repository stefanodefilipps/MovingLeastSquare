import matplotlib.pyplot as plt
import numpy as np

def visualize(X_noisy: np.ndarray, X_smooth: np.ndarray, N_smooth: np.ndarray, A: float, sigma: float):

    # --------------------- Visualizations ---------------------
    # 1) Original noisy shape (3D scatter)
    fig1 = plt.figure(figsize=(6,6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(X_noisy[:,0], X_noisy[:,1], X_noisy[:,2], s=2)
    ax1.set_title("Original Noisy Shape (scatter)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_box_aspect([1,1,0.6])
    plt.show()

    # 2) MLS-smoothed shape (3D scatter)
    fig2 = plt.figure(figsize=(6,6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X_smooth[:,0], X_smooth[:,1], X_smooth[:,2], s=2)
    ax2.set_title("MLS-Smoothed Shape (scatter)")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_box_aspect([1,1,0.6])
    plt.show()

    # 3) Cross-section comparison along y≈0
    mask = np.abs(X_noisy[:,1]) < 0.01
    xs = X_noisy[mask,0]
    z_noisy_section = X_noisy[mask,2]
    z_smooth_section = X_smooth[mask,2]
    z_true_section = A * np.exp(-(xs**2) / (2*sigma**2))

    order = np.argsort(xs)
    xs = xs[order]
    z_noisy_section = z_noisy_section[order]
    z_smooth_section = z_smooth_section[order]
    z_true_section = z_true_section[order]

    fig3 = plt.figure(figsize=(6,4))
    plt.plot(xs, z_noisy_section, '.', label="noisy")
    plt.plot(xs, z_smooth_section, '-', label="MLS")
    plt.plot(xs, z_true_section, '--', label="ground truth")
    plt.title("Cross-section (y≈0): noisy vs MLS vs ground truth")
    plt.xlabel("x"); plt.ylabel("z")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

    # 4) Normals quiver on a subsample
    step = 80
    Qp = X_smooth[::step]
    Qn = N_smooth[::step]
    fig4 = plt.figure(figsize=(6,6))
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.scatter(Qp[:,0], Qp[:,1], Qp[:,2], s=6)
    ax4.quiver(Qp[:,0], Qp[:,1], Qp[:,2], Qn[:,0], Qn[:,1], Qn[:,2], length=0.06, normalize=True)
    ax4.set_title("Estimated Normals on MLS-Smoothed Shape")
    ax4.set_xlabel("X"); ax4.set_ylabel("Y"); ax4.set_zlabel("Z")
    ax4.set_box_aspect([1,1,0.6])
    plt.show()

    # Denoising magnitude
    rms_disp = np.sqrt(np.mean(np.sum((X_noisy - X_smooth)**2, axis=1)))

    print(f"RMS displacement between noisy and smoothed points: {rms_disp:.5f}")