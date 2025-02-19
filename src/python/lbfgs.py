import numpy as np
from scipy.optimize import minimize
from itertools import permutations, combinations
import math
import time

def make_rand_M(k, n, max_clique_size = 6, density=0.05):
    M = np.zeros((n, n, n))

    # Create max clique
    max_clique = np.array([0, 1, 2, 3, 4, 5])
    # max_clique = np.random.choice(n, max_clique_size, replace=False)
    print("max clique actual", np.sort(max_clique))

    # Add clique to matrix
    idxs = permutations(max_clique, k)
    for idx in idxs:
        idx = list(idx)
        M[idx[0], idx[1], idx[2]] = 1.0

    # # Add random edges
    # all_edges = np.array(list(combinations(np.arange(0, n), k)))
    # rand_idxs = np.random.choice(len(all_edges), int(np.round(density*len(all_edges))), replace=False)
    # for edge_idx in rand_idxs:
    #     idxs = permutations(all_edges[edge_idx], k)
    #     for idx in idxs:
    #         M[idx[0], idx[1], idx[2]] = 1.0

    M /= np.max(M)

    # Identity on diagonal
    for i in range(n):
        M[i, i, i] = 1.0

    return M

def get_top_n_indices(x, omega):
    if omega > len(x):
        raise ValueError("omega cannot be greater than the length of the vector.")
    
    return np.sort(list(np.argsort(x)[-omega:][::-1]))


def rayleigh_quotient(A, x):
    """Computes the Rayleigh quotient for a k-th order tensor A and vector x."""
    result = A
    for _ in range(A.ndim):  # Contract tensor along all modes
        result = np.tensordot(result, x, axes=([0], [0]))
    return result

def objective(y, A):
    """Negative Rayleigh quotient (since we maximize, but scipy minimizes)."""
    x = y / np.linalg.norm(y)  # Enforce unit norm
    return -rayleigh_quotient(A, x)

def add_clique_penalty(A, d):
    """Modifies tensor A with clique penalty term."""
    C = (A == 0).astype(int)  # Identify inconsistent pairs
    return A - d * C

def tensor_contract(A, x, k):
    """Contracts tensor A with vector x k times."""
    result = A
    for _ in range(k):
        result = np.tensordot(result, x, axes=([0], [0]))
    return result
def maximize_rayleigh(A, d=1.0, y0=None, max_iter=10):
    """Finds the maximal Rayleigh quotient eigenvalue using L-BFGS-B with dynamic penalty update and nonnegativity constraint."""
    n = A.shape[0]
    if y0 is None:
        y0 = np.abs(np.random.randn(n))  # Initialize with nonnegative values
    
    bounds = [(0, None)] * n  # Enforce x >= 0
    
    for _ in range(max_iter):
        A_d = add_clique_penalty(A, d)
        res = minimize(objective, y0, args=(A_d,), method='L-BFGS-B', bounds=bounds)
        x_opt = np.maximum(res.x, 1e-8)  # Ensure nonnegativity
        x_opt /= np.linalg.norm(x_opt)  # Project back to unit sphere
        
        # Compute Mu and penalty update
        Mu = tensor_contract(A, x_opt, A.ndim - 1)
        Cbu = np.sum(A == 0, axis=0)  # Count number of inconsistent pairs
        idxD = Cbu > 0
        
        if np.any(idxD):
            num = Mu[idxD]
            den = Cbu[idxD]
            deltad = np.mean(np.abs(num / den))
            d += deltad  # Increase penalty
        else:
            break
    
    return x_opt, -res.fun  # Return eigenvector and maximum eigenvalue

# Example usage
n = 10  # Dimension
k = 4   # Order
d = 1.0  # Initial penalty parameter
A = np.random.randn(*([n] * k))  # Random symmetric tensor
x_opt, lambda_max = maximize_rayleigh(A, d)

print("Max eigenvalue:", lambda_max)
print("Eigenvector:", x_opt)
print("Clique found ", get_top_n_indices(x_opt, 6))
