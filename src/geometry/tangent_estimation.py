import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class TangentBundleEstimator:
    """
    Estimates the tangent space T_p M at each point p using Local PCA.
    """
    
    def __init__(self, n_neighbors=30, intrinsic_dim=2):
        self.k = n_neighbors
        self.d = intrinsic_dim
        self.nn_engine = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X):
        """
        X: High-dimensional data (N, D)
        Computes the k-nearest neighbors for every point.
        """
        self.nn_engine.fit(X)
        return self

    def estimate_bases(self, X):
        """
        For each point x_i, returns a basis matrix V_i (D x d) 
        spanning the approximate tangent space.
        """
        N, D = X.shape
        # Find neighbors for all points
        distances, indices = self.nn_engine.kneighbors(X)
        
        tangent_bases = np.zeros((N, D, self.d))
        
        for i in range(N):
            # 1. Get local neighborhood
            neighbor_indices = indices[i]
            local_cloud = X[neighbor_indices]
            
            # 2. Center the neighborhood
            local_mean = np.mean(local_cloud, axis=0)
            centered = local_cloud - local_mean
            
            # 3. Perform PCA (SVD) to find top d principal components
            # These vectors span the tangent plane
            pca = PCA(n_components=self.d)
            pca.fit(centered)
            
            # The components_ are the basis vectors (d, D) -> transpose to (D, d)
            tangent_bases[i] = pca.components_.T
            
        return tangent_bases
