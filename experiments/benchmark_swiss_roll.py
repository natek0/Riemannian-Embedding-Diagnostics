import sys
import os
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geometry.tangent_estimation import TangentBundleEstimator
from src.geometry.jacobian import JacobianEstimator
from src.metrics.distortion import DistortionAnalyzer
from src.viz.heatmaps import plot_distortion

def main():
    print("--- Riemannian Embedding Diagnostics ---")
    
    # 1. Generate Data (Swiss Roll)
    print("1. Generating Swiss Roll...")
    n_points = 1500
    X, t = make_swiss_roll(n_samples=n_points, noise=0.1)
    # X is (1500, 3)
    
    # 2. Embed Data (Using Isomap as the 'Good' candidate)
    print("2. Embedding with Isomap...")
    embedder = Isomap(n_neighbors=15, n_components=2)
    Y = embedder.fit_transform(X)
    
    # 3. Estimate Tangent Spaces (High-D Geometry)
    print("3. Estimating Tangent Spaces...")
    # Swiss roll is a 2D surface in 3D, so intrinsic_dim=2
    tan_est = TangentBundleEstimator(n_neighbors=30, intrinsic_dim=2)
    tan_est.fit(X)
    bases = tan_est.estimate_bases(X)
    
    # 4. Compute Jacobians (The Map F: M -> R^2)
    print("4. Computing Numerical Jacobians...")
    jac_est = JacobianEstimator(n_neighbors=30)
    jacobians = jac_est.compute_jacobians(X, Y, bases)
    
    # 5. Calculate Metrics
    print("5. Calculating Distortion Metrics...")
    analyzer = DistortionAnalyzer()
    area, cond = analyzer.compute_metrics(jacobians)
    
    print(f"Mean Area Distortion: {np.mean(area):.4f} (Ideal = 1.0)")
    print(f"Mean Condition Num:   {np.mean(cond):.4f} (Ideal = 1.0)")
    
    # 6. Visualize
    print("6. Generating Heatmaps...")
    plot_distortion(Y, area, title="Isomap Area Distortion")
    plot_distortion(Y, cond, title="Isomap Shape Distortion (Condition Number)")
    
    print("\nDone! Check the .png files.")

if __name__ == "__main__":
    main()
