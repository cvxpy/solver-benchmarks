import cvxpy as cp
import numpy as np

class NearestCorrelationMatrix:
    """
    Benchmark problem: Nearest Correlation Matrix (NCM).
    
    This problem finds the nearest valid correlation matrix (Positive Semidefinite, 
    unit diagonal) to a given input matrix G, minimizing the Frobenius norm.
    
    This is a standard problem in quantitative finance (Higham, 2002) used 
    to repair empirically estimated or manually adjusted correlation matrices
    that are not mathematically valid (not PSD).
    """
    
    def __init__(self, n, seed=1):
        """
        Args:
            n (int): The dimension of the matrix.
            seed (int): Random seed for reproducibility.
        """
        self.n = n
        self.seed = seed
        self.name = "Nearest Correlation Matrix (Finance)"
        
    def create_problem(self):
        # 1. Set the seed
        np.random.seed(self.seed)
        
        # 2. Generate "Synthetic" Financial Data
        # We start with a valid random correlation matrix structure
        A = np.random.randn(self.n, self.n)
        # A^T * A is always PSD
        base_matrix = A.T @ A 
        
        # Normalize it to be a correlation matrix
        d = np.sqrt(np.diag(base_matrix))
        # Avoid division by zero
        d[d == 0] = 1.0
        valid_G = base_matrix / np.outer(d, d)
        
        # 3. "Break" the matrix
        # In finance, managers tweak numbers or data is missing.
        # We simulate this by adding random noise, which destroys the PSD property.
        # This makes the matrix invalid for risk calculations.
        noise = np.random.normal(0, 0.2, size=(self.n, self.n))
        noise = (noise + noise.T) / 2  # Keep noise symmetric
        
        G_broken = valid_G + noise
        np.fill_diagonal(G_broken, 1.0) # Reset diagonal to 1
        
        # 4. Define the CVXPY Optimization Problem
        # Variable X must be symmetric
        X = cp.Variable((self.n, self.n), symmetric=True)
        
        # Objective: Minimize Sum of Squares error (Frobenius norm squared)
        # Using sum_squares is more efficient for solvers than norm
        objective = cp.Minimize(cp.sum_squares(X - G_broken))
        
        constraints = [
            X >> 0,              # Constraint 1: Must be Positive Semidefinite (PSD)
            cp.diag(X) == 1      # Constraint 2: Unit diagonal
        ]
        
        problem = cp.Problem(objective, constraints)
        
        return problem