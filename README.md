# Moving Least Squares (MLS) for Point Clouds

This repository implements **Moving Least Squares (MLS)** smoothing for 3D point clouds on both **CPU (NumPy/SciPy)** and **GPU (PyTorch)**.  
It supports:
- k-nearest neighbors (kNN) and radius-based neighborhoods
- Polynomial surface fitting of degree 1 (plane) or 2 (quadratic patch)
- Normal estimation
- Batching on GPU to handle large point clouds
- Visualization via [Open3D](http://www.open3d.org/) or matplotlib

---

## 📖 What is Moving Least Squares?

The **MLS algorithm** is a method to project noisy point cloud data onto a smooth surface approximation.  

Given a point cloud:

1. **Neighborhood selection**:  
   For each query point, collect its neighbors (kNN or radius search).

2. **Local reference frame (PCA)**:  
   Perform PCA on the neighborhood to define a tangent plane.

3. **Polynomial fitting**:  
   Fit a polynomial surface  
   \[
   z = f(x,y)
   \]  
   of degree 1 or 2 to the neighbor points, weighted by a Gaussian kernel.

4. **Projection**:  
   Project the query point onto the fitted surface.  
   Optionally, compute the **normal** as the gradient of the fitted polynomial.

The result is a **smoothed point cloud** with optional normals, suitable for visualization, surface reconstruction, or mesh generation.

---

## 📂 Repository Structure

