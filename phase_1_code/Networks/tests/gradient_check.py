import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import empirical_covariance

# Your original data generation here...
# ... (your data generation code)

# Assume p is the number of features, and S is your empirical covariance matrix
p = 10
np.random.seed(42)
random_matrix = np.random.rand(p, p)
S = np.dot(random_matrix, random_matrix.transpose())

# Your objective function
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

# Assume prior_matrix is a zero matrix (no prior knowledge)
prior_matrix = np.zeros((p, p))



# Assume lambda_np and lambda_wp are 0.5
lambda_np = lambda_wp = 0.5

# Set initial guess as the inverse of the empirical covariance matrix
initial_precision_vector = np.linalg.inv(S).flatten()

# Function to compute numerical gradient
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

# Compute the numerical gradient
numerical_grad = numerical_gradient(
    lambda precision_vector: objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix),
    initial_precision_vector
)

##  Print numerical gradient
# print(numerical_grad)

def callback(precision_vector):
    numerical_grad = numerical_gradient(
        lambda precision_vector: objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix),
        precision_vector
    )
    print(f'Numerical Gradient at iteration: {numerical_grad}')

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