import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import empirical_covariance
import networkx as nx
from scipy.linalg import block_diag, eigh, inv
from numpy.random import multivariate_normal


def objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix):
    precision_matrix = precision_vector.reshape((p, p))
    det_value = np.linalg.det(precision_matrix)
    if det_value <= 0 or np.isclose(det_value, 0):
        return np.inf  # return a high cost for non-invertible matrix
    
    log_det = np.log(det_value)
    trace_term = np.trace(np.dot(S, precision_matrix))
    base_objective = -log_det + trace_term

    prior_entries = prior_matrix != 0
    non_prior_entries = prior_matrix == 0
    penalty_wp = lambda_wp * np.sum(np.abs(precision_matrix[prior_entries]))
    penalty_np = lambda_np * np.sum(np.abs(precision_matrix[non_prior_entries]))

    # create a general pentalty term, which is the l1-norm of the precision matrix
    penalty = 0.2 * np.sum(np.abs(precision_matrix))

    objective_value = base_objective + penalty # + penalty_np + penalty_wp
    return objective_value

def numerical_gradient(f, x, epsilon=1e-8):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        unit_vector = np.zeros_like(x, dtype=float)
        unit_vector[i] = 1.0
        
        # Compute function value at x + epsilon * unit_vector
        f_plus = f(x + epsilon * unit_vector)
        
        # Compute function value at x - epsilon * unit_vector
        f_minus = f(x - epsilon * unit_vector)
        
        # Numerical gradient for dimension i
        grad[i] = (f_plus - f_minus) / (2 * epsilon)

    return grad

def callback(precision_vector):
    numerical_grad = numerical_gradient(
        lambda precision_vector: objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix),
        precision_vector
    )
    print(f'Numerical Gradient at iteration: {numerical_grad}')

# PARAMETERS
p = 10
n = 250

# TRUE NETWORK
G = nx.barabasi_albert_graph(p, 1, seed=1)
adj_matrix = nx.to_numpy_array(G)

# PRECISION MATRIX
precision_matrix = -0.5 * adj_matrix

# Add to the diagonal to ensure positive definiteness
# Set each diagonal entry to be larger than the sum of the absolute values of the off-diagonal elements in the corresponding row
diagonal_values = 2 * np.abs(precision_matrix).sum(axis=1)
np.fill_diagonal(precision_matrix, diagonal_values)

# Check if the precision matrix is positive definite
# A simple check is to see if all eigenvalues are positive
eigenvalues = np.linalg.eigh(precision_matrix)[0]
is_positive_definite = np.all(eigenvalues > 0)

# Compute the scaling factors for each variable (square root of the diagonal of the precision matrix)
scaling_factors = np.sqrt(np.diag(precision_matrix))
# Scale the precision matrix
adjusted_precision = np.outer(1 / scaling_factors, 1 / scaling_factors) * precision_matrix

covariance_mat = inv(adjusted_precision)

# PRIOR MATRIX
prior_matrix = np.zeros((p, p))
for i in range(p):
    for j in range(i, p):
        if adj_matrix[i, j] != 0 and np.random.rand() < 0.95 :
            prior_matrix[i, j] = 0.9
            prior_matrix[j, i] = 0.9
        elif adj_matrix[i, j] == 0 and np.random.rand() < 0.05:
            prior_matrix[i, j] = 0.9
            prior_matrix[j, i] = 0.9
np.fill_diagonal(prior_matrix, 0)

# DATA MATRIX
data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)

# Set initial guess as the inverse of the empirical covariance matrix
initial_precision_vector = adjusted_precision.flatten()

# Compute the numerical gradient
numerical_grad = numerical_gradient(
    lambda precision_vector: objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix),
    initial_precision_vector
)

##  Print numerical gradient
# print(numerical_grad)

# Optimization with callback to check gradient at each iteration
result = minimize(
    objective,
    initial_precision_vector,
    args=(S, lambda_np, lambda_wp, prior_matrix),
    method='L-BFGS-B',
    callback=callback,
    options={'disp': True}  # Set to True to display convergence messages
)

# Print optimization result
print(result)