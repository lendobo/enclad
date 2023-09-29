import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import empirical_covariance

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

    # # create a general pentalty term, which is the l1-norm of the precision matrix
    # UNCOMMENT if single penalization term is to be used penalty = 0.05 * np.sum(np.abs(precision_matrix)) 

    objective_value = base_objective + penalty_np + penalty_wp

    return objective_value


#################### COMPUTATIONS ###############################################
# P = number of nodes
p = 10

# Empricial covariance matrix
np.random.seed(42)
random_matrix = np.random.rand(p, p)
S = np.dot(random_matrix, random_matrix.transpose())

prior_matrix = np.zeros((p, p))
# select random combinations of indeces symetrically from the prior matrix
for i in range(p):
    for j in range(i, p):
        # randomly select 0 or 1
        prior_matrix[i, j] = np.random.randint(2)
        # set the corresponding entry in the lower triangle
        prior_matrix[j, i] = prior_matrix[i, j]
# set diagonal to 0
np.fill_diagonal(prior_matrix, 0)

# Set both lambdas
lambda_np = 0.05   # penalize the non-prior matrix more
lambda_wp = 0.0001 # penalize the prior matrix less

# Set initial guess as the inverse of the empirical covariance matrix
initial_precision_vector = np.linalg.inv(S).flatten()

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


