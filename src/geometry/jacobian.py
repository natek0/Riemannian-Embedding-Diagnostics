import numpy as np
from sklearn.linear_model import LinearRegression

class JacobianEstimator:
    """
    Estimates the Jacobian J_i of the embedding map F: M -> R^d
    at each point x_i.
    
    J_i maps tangent vectors in high-D to tangent vectors in low-D.
    """
    
    def __init__(self, n_neighbors=30):
        self.k = n_neighbors

    def compute_jacobians(self, X, Y, tangent_bases):
        """
        X: High-D data (N, D)
        Y: Low-D embedding (N, d_embed)
        tangent_bases: (N, D, d_intrinsic) - The basis of T_p M
        
        Returns: Jacobians (N, d_embed, d_intrinsic)
        """
        from sklearn.neighbors import NearestNeighbors
        
        N = X.shape[0]
        d_out = Y.shape[1]
        d_in = tangent_bases.shape[2]
        
        nn = NearestNeighbors(n_neighbors=self.k)
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        
        jacobians = np.zeros((N, d_out, d_in))
        
        for i in range(N):
            neighbor_idx = indices[i]
            
            # 1. Get vectors in High-D and Low-D relative to center x_i, y_i
            dX = X[neighbor_idx] - X[i] # (k, D)
            dY = Y[neighbor_idx] - Y[i] # (k, d_out)
            
            # 2. Project High-D vectors onto the Tangent Basis V_i
            # This converts local coords from R^D to R^d_intrinsic
            # V_i is (D, d_in)
            V_i = tangent_bases[i]
            
            # Local coordinates U = dX * V_i 
            # (k, D) @ (D, d_in) -> (k, d_in)
            U = dX @ V_i
            
            # 3. Solve Linear System: dY approx U @ J.T
            # We want J such that dY = U J^T
            # Use Least Squares regression
            try:
                # LinearRegression fits y = Xw + b. Here 'X' is U, 'y' is dY.
                reg = LinearRegression(fit_intercept=False)
                reg.fit(U, dY)
                
                # coef_ is (n_targets, n_features) -> (d_out, d_in)
                jacobians[i] = reg.coef_
            except:
                # Fallback for degenerate neighborhoods
                jacobians[i] = np.eye(d_out, d_in)
                
        return jacobians
