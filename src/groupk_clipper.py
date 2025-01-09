import numpy as np
import cvxpy as cp
from ncpol2sdpa import generate_variables, SdpRelaxation
from itertools import permutations, combinations
import math
from path import get_M

np.random.seed(800) # Different clique sizes do not converge correctly
# np.random.seed(0) 

def calculate_k_norm(x, k):
    """
    Calculate the k-norm of an array.

    Parameters:
    x (numpy.ndarray): Input array.
    k (int): The norm degree, must be greater than 0.

    Returns:
    float: The k-norm of the array.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    
    # Compute the sum of absolute values raised to the power of k
    sum_of_powers = np.sum(np.abs(x) ** k)

    knorm = sum_of_powers ** (1.0 / k)
    # print("K NORM", knorm, x)
    
    # Return the k-norm
    return knorm

def tensor_vector_multiplication(A, X, k):
    """
    Perform the tensor-vector multiplication as described.

    Parameters:
    A (ndarray): A tensor of shape (n, n, ..., n) with `k` dimensions.
    X (ndarray): A vector of shape (n,).
    k (int): Number of tensor contractions along axes.

    Returns:
    float: Result of the tensor-vector multiplication.
    """
    """
    Perform the tensor-vector multiplication manually with loops.

    Parameters:
    A (ndarray): A tensor of shape (n, n, ..., n) with `k` dimensions.
    X (ndarray): A vector of shape (n,).
    k (int): Number of tensor contractions along axes.

    Returns:
    float: Result of the tensor-vector multiplication.
    """
    n = A.shape[0]  # Assuming the tensor is of shape (n, n, ..., n)
    
    # Perform the contractions step by step
    for contraction in range(k):
        # Compute the contraction of the last axis of the tensor with the vector
        new_shape = A.shape[:-1]  # The resulting shape after contraction
        result = np.zeros(new_shape)  # Initialize a tensor for the result
        
        # Perform the summation over the last axis
        for indices in np.ndindex(new_shape):
            for i in range(n):
                result[indices] += A[(*indices, i)] * X[i]
        
        # Update A to the contracted result
        A = result
    
    # After k contractions, A becomes a scalar
    return A


def z_eigenvalue(tensor, max_iter=5000, tol=1e-10):
    """
    Compute the dominant H-eigenvalue and eigenvector of a symmetric even-order tensor.

    Parameters:
    - tensor: A symmetric even-order tensor (numpy ndarray).
    - max_iter: Maximum number of iterations.
    - tol: Convergence tolerance.

    Returns:
    - eigenvalue: Dominant H-eigenvalue.
    - eigenvector: Corresponding H-eigenvector.
    """
    converged_flag = False
    # Ensure tensor is symmetric and even-order
    order = len(tensor.shape)
    if order % 2 != 0:
        raise ValueError("Tensor must have even order.")

    n = tensor.shape[0]  # Dimension of the tensor
    vec = np.random.rand(n)  # Random initial vector
    vec = vec / np.linalg.norm(vec)  # Normalize the vector

    for _ in range(max_iter):
        # Compute tensor contraction: \mathcal{A} x^{m-1}
        new_vec = np.tensordot(tensor, vec, axes=([1], [0]))  # Contract first axis
        for _ in range(order - 2):  # Contract remaining axes
            new_vec = np.tensordot(new_vec, vec, axes=([0], [0]))

        new_vec /= np.linalg.norm(new_vec)  # Normalize the resulting vector

        # Check for convergence
        if np.linalg.norm(new_vec - vec) < tol:
            converged_flag = True
            break

        vec = new_vec

    if converged_flag:
        print("\nZ Power Converged")
    else:
        print("\nZ Power Not Converged")

    # Compute H-eigenvalue: \lambda = \mathcal{A} x^{m}
    eigenvalue = tensor
    for _ in range(order):
        eigenvalue = np.tensordot(eigenvalue, vec, axes=([0], [0]))

    return eigenvalue, vec

def verify_z_eigenvalue(tensor, eigenvalue, eigenvector, tol=1e-6):
    """
    Verify if the computed eigenvalue and eigenvector satisfy the H-eigenvalue definition.

    Parameters:
    - tensor: Symmetric tensor (numpy ndarray).
    - eigenvalue: Computed H-eigenvalue.
    - eigenvector: Computed H-eigenvector.
    - tol: Tolerance for numerical comparison.

    Returns:
    - is_valid: True if the eigenvalue and eigenvector are valid, False otherwise.
    """
    # Compute the order of the tensor
    order = len(tensor.shape)

    # Check normalization of the eigenvector
    if not np.isclose(np.linalg.norm(eigenvector), 1, atol=tol):
        print("Eigenvector is not normalized.")
        return False

    # Compute \mathcal{A} x^{m-1}
    result = tensor
    for _ in range(order - 1):
        result = np.tensordot(result, eigenvector, axes=([0], [0]))

    # Check if \mathcal{A} x^{m-1} ≈ λx
    lhs = result  # Left-hand side: \mathcal{A} x^{m-1}
    rhs = eigenvalue * eigenvector  # Right-hand side: λx
    print("lhs", lhs)
    print("rhs", rhs)
    if not np.allclose(lhs, rhs, atol=tol):
        print("H-eigenvalue equation is not satisfied.")
        return False

    print("H-eigenvalue and eigenvector are valid.")
    return True

def h_eigenvalue(tensor, max_iter=1000, tol=1e-6):
    """
    Compute the dominant H-eigenvalue and H-eigenvector of a symmetric tensor
    using the tensor power method.
    """
    converged_flag = False

    order = len(tensor.shape)
    n = tensor.shape[0]
    x = np.random.rand(n)  # Initial random vector
    x = x / np.linalg.norm(x)  # Normalize the vector

    for _ in range(max_iter):
        # Compute A x^(m-1) element-wise
        result = tensor
        for _ in range(order - 1):
            result = np.tensordot(result, x ** (order - 1), axes=([0], [0]))
        
        # Normalize the result
        norm = np.linalg.norm(result)
        x_new = result / norm
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            converged_flag = True
            break
        
        x = x_new

    if converged_flag:
        print("\nH Power Converged")
    else:
        print("\nH Power Not Converged")

    # Compute the H-eigenvalue
    # print("num", np.tensordot(tensor, x ** (order - 1), axes=([0], [0])) )
    # print("denom", x ** (order - 1))
    eigenvalue = np.sum(np.tensordot(tensor, x ** (order - 1), axes=([0], [0])) * (x ** (order - 1)))

    return eigenvalue, x

def verify_h_eigenvalue(tensor, eigenvalue, eigenvector):
    """
    Verify the H-eigenvalue condition by comparing the LHS and RHS of the equation:
    (A x^{m-1})_i = λ x_i^{m-1}

    Parameters:
    - tensor: Symmetric tensor (numpy ndarray).
    - eigenvalue: The H-eigenvalue from the h_eigenvalue function.
    - eigenvector: The H-eigenvector from the h_eigenvalue function.

    Returns:
    - is_valid: Boolean indicating whether the LHS and RHS are approximately equal.
    """
    # Compute the LHS: (A x^{m-1})
    order = len(tensor.shape)
    lhs = tensor
    for _ in range(order - 1):
        lhs = np.tensordot(lhs, eigenvector, axes=([0], [0]))
    
    # Compute the RHS: λ x^{m-1}
    rhs = eigenvalue * np.power(eigenvector, order-1)
    print("lhs", lhs)
    print("rhs", rhs)

    # Compare LHS and RHS
    is_valid = np.allclose(lhs, rhs, atol=1e-6)  # Use a tolerance for numerical precision

    # Print the result and verification
    if is_valid:
        print(f"The H-eigenvalue λ = {eigenvalue} is valid.")
    else:
        print(f"The H-eigenvalue λ = {eigenvalue} is NOT valid.")
    
    return is_valid

def get_Axk(A, x, k):
    # Gives A x ^ k
    result = A
    for _ in range(k):
        result = np.tensordot(result, x, axes=([0], [0]))

    return result

def h_eigenvalue_rayleigh(M, k, x_init = None, d = 0, max_iter=50, tol=1e-6, step_size=0.01, alpha=1e-4, beta=0.5):
    """
    Compute the dominant H-eigenvalue and H-eigenvector for a tensor using gradient ascent,
    assuming the denominator of the Rayleigh quotient is 1.

    Parameters:
        tensor (np.ndarray): Symmetric tensor of order k and size (n, n, ..., n).
        k (int): Order of the tensor.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        step_size (float): Learning rate for gradient ascent.

    Returns:
        float: Dominant H-eigenvalue.
        np.ndarray: Corresponding H-eigenvector.
    """

    converged_flag = False
    n = M.shape[0]
    if x_init is None:
        x = np.random.rand(n)  # Initialize random vector
    else:
        x = x_init.astype(float) 
        # x = np.array([2., 0.5, 0.01])
    x = np.maximum(x, 0) # project onto positive orthant
    print("k norm", calculate_k_norm(x, k) )
    x /= calculate_k_norm(x, k).astype(float)  #np.linalg.norm(x)  # Normalize initial vector

    C = M
    Md = M + d*C



    def rayleigh_quotient(A, x):
        result = A
        # Contract the tensor k times for the numerator
        for _ in range(k):
            result = np.tensordot(result, x, axes=([0], [0]))
        return result

    def gradient(x, A):
        # Contract the tensor (k-1) times for the gradient k A ^(k-1)
        result = A
        for _ in range(k - 1):
            result = np.tensordot(result, x, axes=([0], [0]))

        print("\ngrad 1", result)
        ones = np.ones_like(x)
        result = k * result + d * x - d * ones * np.sum(x)
        return k * result

    for _ in range(max_iter):
        # print("\nx ", x)

        # Gradient at the current point
        grad = gradient(x, Md)

        # Backtracking line search to find step size
        t = 1.0  # Initial step size
        f_x = rayleigh_quotient(M, x)

        while rayleigh_quotient(M, x + t * grad) < f_x + alpha * t * np.dot(grad, grad):
            t *= beta
        

        # Update x using the computed step size
        x_new = x + alpha * grad
        x_new = np.maximum(x_new, 0) # project onto positive orthant
        k_norm = calculate_k_norm(x_new, k)


        if k_norm > 0:
            x_new /=  k_norm #np.linalg.norm(x_new)  # Normalize to unit norm
        else:
            print("CANNOT NORMALIZE")
            exit()
        # print("x normalized", x_new)

        # Check for convergence
        # print("rho new vs prev", rayleigh_quotient(M, x_new) , rayleigh_quotient(M, x))
        if abs(rayleigh_quotient(M, x_new) - rayleigh_quotient(M, x)) < tol:
        # if np.linalg.norm(x_new - x) < tol:
            converged_flag = True
            break

        print("\nx old", x)
        x = x_new

        print("grad", grad)
        print("rayleigh", f_x)
        print("k norm", k_norm)
        print("x new", x_new)
        print("next x old", x)

    if converged_flag:
        print("\nH Rayleigh Converged")
    else:
        print("\nH Rayleigh Not Converged")

    # Compute final eigenvalue
    eigenvalue = rayleigh_quotient(M, x)
    return eigenvalue, x
    
def get_top_n_indices(x, omega):
    """
    Get the indices of the n highest values in a vector.

    Args:
        x (list or numpy array): The input vector.
        n (int): The number of highest values to retrieve indices for.

    Returns:
        list: Indices of the n highest values, sorted by their order in x.
    """
    if omega > len(x):
        raise ValueError("omega cannot be greater than the length of the vector.")
    
    return list(np.argsort(x)[-omega:][::-1])

def create_c_from_m(M):
    """
    Create an array C of the same dimensions as M, 
    where C has 1s wherever M has 0s, and 0s elsewhere.

    Args:
        M (numpy array): Input numpy array.

    Returns:
        numpy array: Array C with 1s at positions where M has 0s.
    """
    C = np.zeros_like(M)
    C[M == 0] = 1
    return C

def get_new_d(u, C, M, d_prev, eps=1e-9):
    """
    Perform the update calculations equivalent to the provided C++ code.

    Parameters:
    u (numpy.ndarray): The input vector.
    C (numpy.ndarray): The constraint matrix.
    M (numpy.ndarray): The affinity matrix.
    ones (numpy.ndarray): A vector of ones of the same size as `u`.
    eps (float): A small threshold value.
    d (float): The value to be updated.

    Returns:
    float: The updated value of `d`.
    """

    # Initialize
    d = d_prev
    k = M.ndim

    # Calculate Cbu
    ones = np.ones_like(u)
    Cbu = ones * np.sum(u) - get_Axk(C, u, k - 1) - u
    print("Cbu", Cbu)
    print("u", u)

    # Calculate idxD as a boolean mask
    idxD = (Cbu > eps) & (u > eps)

    print("\nCurr sum", np.sum(idxD))
    if np.sum(idxD) > 0:

        # Calculate Mu
        Mu = get_Axk(M, u, k - 1)

        # Select elements from Mu and Cbu using the idxD mask
        num = Mu[idxD]
        den = Cbu[idxD]

        # Calculate deltad
        deltad = np.mean(np.abs(num / den))

        # Update d
        d += deltad
    else:
        d = None

    return d

#-------------------------------
# # Example: 4th-order symmetric tensor
# M = np.array([
#     [[[2, 0], [0, 1]], 
#      [[0, 1], [1, 0]]],
#     [[[0, 1], [1, 0]], 
#      [[1, 0], [0, 3]]]
# ])

#-------------------------------
# M = np.array([
#     [2, 1, 3],
#     [1, 4, 5],
#     [3, 5, 6]
# ])

#-------------------------------
# M = np.array([[1, 2, 3],
#               [2, 3, 4],
#               [3, 4, 1]])
# vals, vecs = np.linalg.eig(M)
# print("Vals actual", vals)
# print("Vecs actual")
# print(vecs)

#-------------------------------
# n= 3
# M = np.zeros((n, n, n, n))
# M[0, 0, 0, 0] = -4
# M[1, 1, 1, 1] = -4
# M[2, 2, 2, 2] = -4

# M[0, 2, 2, 2] = 1
# M[2, 0, 2, 2] = 1
# M[2, 2, 0, 2] = 1
# M[2, 2, 2, 0] = 1

#-------------------------------
# # Different sized cliques, one max clique
# n = 9
# M = np.zeros((n, n, n, n))
# idxs = permutations([0, 1, 2, 3, 4], 4)
# x_tracker = {}
# for idx in idxs:
#     key = tuple(sorted(list(idx)))
#     if key in x_tracker:
#         x_tracker[key]+=1
#     else:
#         x_tracker[key] = 0
#     idx = list(idx)
#     M[idx[0], idx[1], idx[2], idx[3]] = 0.5

# idxs = permutations([5, 6, 7, 8], 4)
# for idx in idxs:
#     key = tuple(sorted(list(idx)))
#     if key in x_tracker:
#         x_tracker[key]+=1
#     else:
#         x_tracker[key] = 0
#     idx = list(idx)
#     M[idx[0], idx[1], idx[2], idx[3]] = 0.5

# M /= np.max(M)

# # Identity on diagonal
# for i in range(n):
#     M[i, i, i, i] = 1
#-------------------------------
# Two max cliques, different weights
# n = 9
# M = np.zeros((n, n, n, n))
# idxs = permutations([0, 1, 2, 3, 4], 4)
# x_tracker = {}
# for idx in idxs:
#     key = tuple(sorted(list(idx)))
#     if key in x_tracker:
#         x_tracker[key]+=1
#     else:
#         x_tracker[key] = 0
#     idx = list(idx)
#     M[idx[0], idx[1], idx[2], idx[3]] = 0.5

# idxs = permutations([4, 5, 6, 7, 8], 4)
# for idx in idxs:
#     key = tuple(sorted(list(idx)))
#     if key in x_tracker:
#         x_tracker[key]+=1
#     else:
#         x_tracker[key] = 0
#     idx = list(idx)
#     M[idx[0], idx[1], idx[2], idx[3]] = 2

# M /= np.max(M)

# # Identity on diagonal
# for i in range(n):
#     M[i, i, i, i] = 1

#-------------------------------
# From CLIPPER example
# M = np.array([[1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
#  [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
#  [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
#  [1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.],
#  [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
#  [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
#  [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
#  [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
#  [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

# ---------------------------------------
# # From CLIPPER example
# M = np.array([
#     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2964, 0.0],
#     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0747, 0.0],
#     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0555, 0.2547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0102, 0.0, 0.7715, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0063, 0.0, 0.3846, 0.0, 0.0003, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0063, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0555, 0.0063, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.9722, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.2547, 0.0, 0.0, 1.0, 0.0, 0.0023, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.3846, 0.0, 0.0, 1.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0023, 0.0001, 1.0, 0.7914, 0.0, 0.0, 0.0, 0.0617, 0.0, 0.0, 0.9938, 0.0, 0.0, 0.0007],
#     [0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.0, 0.7914, 1.0, 0.0, 0.0, 0.0001, 0.0091, 0.0, 0.2503, 0.0222, 0.0549, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008],
#     [0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 1.0, 0.0, 0.9978, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0138, 0.0, 0.0102, 0.0, 0.0, 0.0, 0.0, 0.0617, 0.0091, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9978, 0.0, 1.0, 0.0012, 0.0, 0.0, 0.0, 0.0074],
#     [0.0, 0.0, 0.0, 0.7715, 0.0063, 0.9722, 0.0, 0.0, 0.0, 0.2503, 0.0, 0.0, 0.0, 0.0, 0.0012, 1.0, 0.0026, 0.0217, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9938, 0.0222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026, 1.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0549, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0217, 0.0, 1.0, 0.0007, 0.0],
#     [0.2964, 0.0, 0.0747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 1.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 0.0, 0.0008, 0.0, 0.0, 0.0, 0.0074, 0.0, 0.0, 0.0, 0.0, 1.0]
# ])
#-------------------------------
# # From simulated range measurements
# M = get_M()
# M /= np.max(M)
# print(M)
# exit()

#-------------------------------
# From Brendom groupk example
n = 7

M = np.zeros((n, n, n))
for perm in permutations([0, 1, 3, 4], 3):
    M[perm[0], perm[1], perm[2]] = 1

for perm in permutations([0, 2, 4], 3):
    M[perm[0], perm[1], perm[2]] = 1

for perm in permutations([4, 5, 6], 3):
    M[perm[0], perm[1], perm[2]] = 1

for perm in permutations([3, 4, 6], 3):
    M[perm[0], perm[1], perm[2]] = 1

# Diagonal
for i in range(n):
    M[i, i, i] = 1

# print(M)
# exit()
#---------Cross Check Eigenvalues---------
CROSSCHECK_FLAG = False
k = M.ndim
print("k", k)

if CROSSCHECK_FLAG:
    z_val, z_vec = z_eigenvalue(M)
    print("\nZ Vals Power Method")
    print("Z-Eigenvalue:", z_val)
    print("Z-Eigenvector:", z_vec)

    print("Rayleigh Quotient", tensor_vector_multiplication(M, z_vec, k) / calculate_k_norm(z_vec, k))

    # Verify the results
    is_valid = verify_z_eigenvalue(M, z_val,  z_vec / np.linalg.norm(z_vec))
    print("Is valid Z-eigenvalue:", is_valid)


    h_val, h_vec = h_eigenvalue(M)
    print("\nH Vals Power Method")
    print("\nH-Eigenvalue:", h_val)
    print("H-Eigenvector:", h_vec)

    print("Rayleigh Quotient", tensor_vector_multiplication(M, h_vec, k) / calculate_k_norm(h_vec, k))

    # Verify the results
    is_valid = verify_h_eigenvalue(M, h_val, h_vec / np.linalg.norm(h_vec))
    print("Is valid H-eigenvalue:", is_valid)


# -----------Initial Calculation Method---------
A_test = np.ones((3,3))
x_test = np.array([1, 2, 3])
print(A_test*x_test)
print(np.sum(A_test*x_test, axis=(0)))
print(np.tensordot(A_test, x_test, axes=([0], [0])))
exit()

eigenvalue, eigenvector = h_eigenvalue_rayleigh(M, k, x_init=np.ones(n), d=0)
print("\nH Vals Rayleigh Quotient")
print("H-eigenvalue:", eigenvalue)
print("H-eigenvector:", eigenvector)
exit()
print("\nVerify")
is_valid = verify_h_eigenvalue(M, eigenvalue, eigenvector / np.linalg.norm(eigenvector))
print("Is valid H-eigenvalue:", is_valid)


print("\nClique Number Upper Bound")
# upper_bound = (eigenvalue*math.factorial(k-1))**(1/(k-1)) + (k-1)
# print(upper_bound)
upper_bound = eigenvalue / math.factorial(k) + k
upper_bound_rounded = int(np.round(upper_bound))
print(upper_bound, upper_bound_rounded)

# Get w highest entries from u
print("\nDensest Clique")
print(get_top_n_indices(eigenvector, upper_bound_rounded))

# exit()

# -------------Begin Gradient Ascent-------------
# Make constraint matrix
C = M#create_c_from_m(M)

# Ensure clique constraints are met by adding penalty term
u = np.copy(eigenvector)
d = get_new_d(u, C, M, d_prev=0) # Penalty term initialized at 0, no active constraints

ones = np.ones_like(u)
eps = 1e-9
rho = None

# print(M)now
# print(C)
# exit()
for i in range(2300):
    print("i", i)

    # num_nonzero = np.count_nonzero(u)
    # if num_nonzero < 5:
    #     exit()
    # else:
    #     print("Num nonzero:", num_nonzero)

    if d is None:
        print("d is None")
        break
    else:
        rho, u = h_eigenvalue_rayleigh(M, k, x_init=u, d=d)
        print("\nH Vals Rayleigh Quotient")
        print("H-eigenvalue:", rho)
        print("H-eigenvector:", u)
        # print("\nVerify")
        # is_valid = verify_h_eigenvalue(M, rho, )
        # print("Is valid H-eigenvalue:", is_valid)

        print("\nClique Number Upper Bound, idx ", i)
        # # upper_bound = (rho*math.factorial(k-1))**(1/(k-1)) + (k-1)
        # # print(upper_bound)
        upper_bound = rho / math.factorial(k) + k
        print(upper_bound)

        # # Get w highest entries from u
        # print("\nDensest Clique")
        # print(get_top_n_indices(u, upper_bound))
        # d = calculate_d(u, C, M, d)

        Cu = C
        for _ in range(k - 1):
            Cu = np.tensordot(Cu, u, axes=([0], [0]))

        Cbu = ones * np.sum(u) - Cu - u
        idxD = ((Cbu > eps) & (u > eps)).astype(int)
        # print(idxD)
        # exit()

        print("\nu", u)
        # print("Cbu", Cbu)
        print("\nCbu eps", (Cbu > eps).astype(int))
        # print("u eps", (u > eps).astype(int))

        # print(idxD)

        print("\nCurr sum", np.sum(idxD))

        # d += 0.001
        d_temp = get_new_d(u, C, M, d_prev=d)
        if d_temp is None:
            # Clique constraints satisfied
            break
        else:
            d = d_temp

print("Final spectral radius:", rho)
upper_bound = int(np.round(rho / math.factorial(k) + k))
print("Final clique number upper bound:", rho / math.factorial(k) + k, upper_bound)
print("Final eigenvector:", u)
print("Densest clique:", get_top_n_indices(u, upper_bound))