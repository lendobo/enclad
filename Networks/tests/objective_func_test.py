import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import empirical_covariance

# Assume p is the number of features
p = 10

# Assume S is your empirical covariance matrix
# For simplicity, let's generate a random positive definite matrix for S
np.random.seed(42)
random_matrix = np.random.rand(p, p)
S = np.dot(random_matrix, random_matrix.transpose())

# Define the objective function
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

    objective_value = base_objective # + penalty # + penalty_np + penalty_wp

    return objective_value

# Assume prior_matrix is a zero matrix (no prior knowledge)
prior_matrix = np.zeros((p, p))

# Assume lambda_np and lambda_wp are 0.5
lambda_np = lambda_wp = 0.5

# Set initial guess as the inverse of the empirical covariance matrix
initial_precision_vector = np.eye(p).flatten() # np.linalg.inv(S).flatten()

# Attempt to minimize the objective function
result = minimize(
    objective,
    initial_precision_vector,
    args=(S, lambda_np, lambda_wp, prior_matrix),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'ftol': 1e-15}
)

# Print the result
print(result)


