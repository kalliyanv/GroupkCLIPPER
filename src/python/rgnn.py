import numpy as np
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

def get_Axk(A, x, k):
    # Gives A x ^ k
    result = A
    for _ in range(k):
        result = np.tensordot(result, x, axes=([0], [0]))
    return result

def compute_lambda1(A, x, k, c):
    Ax_m2 = get_Axk(A, x, k-2)  # A x^(m-2)
    Ax_m1 = get_Axk(Ax_m2, x, 1)  # A x^(m-1)
    Ax_m = np.dot(Ax_m1, x)  # A x^m
    
    return (k - 1) * Ax_m2 - k * x * Ax_m1.T + Ax_m * (2 * np.outer(x, x) - np.eye(len(x))) + c * np.outer(x, x)

def get_eigen(A, x_init, k, alpha=3.0, tol=1e-10, max_iter=1000, c=100):
    # x_init[x_init < 0] = 0
    # x_init = np.maximum(x_init, 0)
    x_k = x_init / np.linalg.norm(x_init)
    
    
    for i in range(max_iter):
        Ax_m1 = get_Axk(A, x_k, k-1)  # A x^(m-1)
        Ax_m = Ax_m1 @ x_k #np.dot(Ax_m1, x_k)  # A x^m
        
        lambda1 = compute_lambda1(A, x_k, k, c)
        x_k1 = x_k - alpha * lambda1.T @ (Ax_m1 - Ax_m * x_k)
        # x_k1[x_k1 < 0] = 0
        # x_k1 = np.maximum(x_k1, 0)
        x_k1 /= np.linalg.norm(x_k1)
        
        if np.linalg.norm(x_k1 - x_k) < tol:
            print(i)
            break
        
        x_k = x_k1
    
    lambda_k = Ax_m1 @ x_k#np.dot(Ax_m1, x_k)  # Final eigenvalue estimate
    return x_k, lambda_k

def get_top_n_indices(x, omega):
    if omega > len(x):
        raise ValueError("omega cannot be greater than the length of the vector.")
    
    return np.sort(list(np.argsort(x)[-omega:][::-1]))

k = 3
n = 20
max_clique_size = 6
M = make_rand_M(k, n, max_clique_size)
u0 = np.abs(np.random.randn(n))

start = time.time()
eigenvec, eigenval = get_eigen(M, u0, k)
end = time.time()

upper_bound = eigenval / math.factorial(k) + k


print("Eigenvec", eigenvec)
print("eigenval", eigenval)
print("Clique found ", get_top_n_indices(eigenvec, max_clique_size))
print("time", end - start)
# print(upper_bound)