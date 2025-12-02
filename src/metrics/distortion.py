import numpy as np

class DistortionAnalyzer:
    """
    Analyzes the Jacobian matrices to quantify geometric distortion.
    """
    
    def compute_metrics(self, jacobians):
        """
        jacobians: (N, d_out, d_in)
        
        Returns:
            area_distortion: (N,) - How much local area/volume changed
            condition_number: (N,) - How much shape was skewed (Shear)
        """
        N = jacobians.shape[0]
        area = np.zeros(N)
        cond_num = np.zeros(N)
        
        for i in range(N):
            J = jacobians[i] # (d_out, d_in)
            
            # Compute Metric Tensor M = J.T @ J (The Pull-back Metric)
            # This tells us how dot products in Low-D relate to High-D
            M = J.T @ J
            
            # Eigenvalues of the metric tensor
            # These represent the squared stretching factors along principal axes
            try:
                eigvals = np.linalg.eigvalsh(M)
                # Filter tiny values to avoid div by zero
                eigvals = np.maximum(eigvals, 1e-10)
                
                # 1. Area Distortion = sqrt(det(M)) = product of singular values
                # If d_in = d_out, this is determinant of J
                area[i] = np.sqrt(np.prod(eigvals))
                
                # 2. Condition Number = sqrt(max_eig / min_eig)
                # Measures anisotropy (how circle maps to ellipse)
                cond_num[i] = np.sqrt(np.max(eigvals) / np.min(eigvals))
                
            except Exception:
                area[i] = 1.0
                cond_num[i] = 1.0
                
        return area, cond_num
