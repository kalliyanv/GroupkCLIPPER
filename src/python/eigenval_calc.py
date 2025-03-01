import numpy as np
from itertools import permutations, combinations
import time
import math

# np.random.seed(0)

# Define the order of the tensor
k = 4  # Even-order tensor (example)
n = 9  # Dimension of the vector v

# Generate a random symmetric tensor M
# M = np.random.randn(n, n, n, n)
# M = (M + M.transpose(1, 0, 2, 3)) / 2  # Symmetrization

M = np.zeros((n, n, n, n))
idxs = permutations([0, 1, 2, 3, 4], k)
x_tracker = {}
for idx in idxs:
    key = tuple(sorted(list(idx)))
    if key in x_tracker:
        x_tracker[key]+=1
    else:
        x_tracker[key] = 0
    idx = list(idx)
    M[idx[0], idx[1], idx[2], idx[3]] = 0.5

idxs = permutations([5, 6, 7, 8], k)
for idx in idxs:
    key = tuple(sorted(list(idx)))
    if key in x_tracker:
        x_tracker[key]+=1
    else:
        x_tracker[key] = 0
    idx = list(idx)
    M[idx[0], idx[1], idx[2], idx[3]] = 0.5

M /= np.max(M)

# Identity on diagonal
for i in range(n):
    M[i, i, i, i] = 1

# Normalize the tensor to avoid instability
# M /= np.linalg.norm(M)

# Define the Rayleigh Quotient function
def rayleigh_quotient(v):
    """Computes R(v) = (v^T M v^(k-1)) / (v^T v)"""
    numerator = np.tensordot(M, np.outer(v, v), axes=k-2) @ v
    denominator = np.dot(v, v)
    return np.dot(v, numerator) / denominator

# Compute the gradient of the Rayleigh quotient
def gradient_rayleigh(v):
    """Computes the gradient ∇R(v)"""
    Rv = rayleigh_quotient(v)
    numerator = np.tensordot(M, np.outer(v, v), axes=k-2) @ v
    return numerator - Rv * v  # Gradient formula

# Project v onto unit sphere
def project(v):
    v[v < 0] = 0
    v = v / np.linalg.norm(v)
    return v

# RQG-NN Solver
def rqg_nn_solver(M, v, max_iters=1000, lr=0.01, tol=1e-6):
    """Solves for the dominant 𝒵-eigenpair using Rayleigh Quotient Gradient NN"""
    v = project(v)  # Normalize initial guess

    for i in range(max_iters):
        grad = gradient_rayleigh(v)
        v_new = v + lr * grad  # Gradient ascent
        v_new = project(v_new)  # Project onto unit sphere

        # Check convergence
        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged in {i} iterations")
            break
        # cos_sim = np.dot(v_new, v) / (np.linalg.norm(v_new) * np.linalg.norm(v))
        # if cos_sim > 0.999:
        #     print(f"Converged in {i} iterations")
        #     break

        v = v_new

    eigenvalue = rayleigh_quotient(v)
    return eigenvalue, v

def get_top_n_indices(x, omega):
    if omega > len(x):
        raise ValueError("omega cannot be greater than the length of the vector.")
    
    return np.sort(list(np.argsort(x)[-omega:][::-1]))

def get_Axk(A, x, k):
    # Gives A x ^ k
    result = A
    for _ in range(k):
        result = np.tensordot(result, x, axes=([0], [0]))

    return result

# Run the solver
d = 0
eps = 1e-9
v = np.random.randn(n)

start_time = time.time()
# lambda_max, eigenvector = rqg_nn_solver(M)
for i in range(10):
    # Run gradient ascent
    lambda_max, eigenvector = rqg_nn_solver(M, v)

    # Increase penalty
    Axk_min_1 = get_Axk(M, v, k-1)
    Cbu = np.ones_like(v)*np.sum(v) - (Axk_min_1 - v) - v
    idxD = ((Cbu > eps) & (v > eps)).astype(int)
    # print("Cbu", (Axk_min_1 - v))
    # print("v", v > eps)
    # print("idxD", idxD)

    # if (np.sum(idxD > 0)):
    #     Mu = Axk_min_1
    #     num = Mu[idxD]
    #     den = Cbu[idxD]
    #     deltad = np.mean(np.abs(num / den))

    #     d += deltad

    #     C = np.zeros_like(M)
    #     C[M > 0] = 1 
    #     M = M + C*d
    #     # print(M)
    #     # print(C)
    #     # M = 

    # else:
    #     break

end_time = time.time()

# Print results
print(f"Dominant eigenvalue: {lambda_max}")
print(f"Corresponding eigenvector: {eigenvector}")

upper_bound = int(np.round(lambda_max / math.factorial(k) + k)) # int(np.round(lambda_max)) #
print("Densest clique:", get_top_n_indices(eigenvector, 5))
print("Time", end_time - start_time)

# eigenval_np, eigenvec_np = np.linalg.eig(M)
# print("\nNp val", max(eigenval_np))
