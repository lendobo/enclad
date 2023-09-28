import numpy as np
import math
from scipy.special import comb
from scipy.optimize import minimize
import scipy.stats as stats
from itertools import combinations
from itertools import product
from sklearn.covariance import empirical_covariance
from concurrent.futures import ProcessPoolExecutor


####### TODO #######
# Vectorize the for loops (advanced)
# implement parallelization for Q * J optimization runs (advanced)
# Once it runs, implement alternative and check if results are still the same
# check if the denominator of eq (12) is implemented correctly
# The function theta_k = post_distro.cdf(zk_vec[:,e_k] + epsilon) - post_distro.cdf(zk_vec[:,e_k] - epsilon), could have the wrong shape

###### Possible issues #######
# Currently, we assume an edge is present if the optimized precision matrix has an absolute value > 1e-5.
# If the resulting network is too dense, we might have to change this threshold. However, the LASSO pushes many values to exactly 0, 
# so this might not be a problem.


def optimize_for_q_and_j(params):
    q, lambdax = params
    all_subs, data, p, prior_matrix = global_params

    sub_sample = data[np.array(all_subs[q]), :]
    S = empirical_covariance(sub_sample)
    initial_precision_vector = np.eye(p).flatten()
    result = minimize(
        objective,  # Replace with your actual objective function
        initial_precision_vector,
        args=(S, lambdax, lambdax, prior_matrix),  # And the actual arguments for your objective function
        method='L-BFGS-B',
    )
    if result.success:
        opt_precision_mat = result.x.reshape((p, p))
        edge_counts = (np.abs(opt_precision_mat) > 1e-5).astype(int)  # Assume edge exists if absolute value > 1e-5
        return q, lambdax, edge_counts
    else:
        return q, lambdax, np.zeros((p, p))

def subsample_optimiser(data, b, Q, lambda_range, prior_matrix):
    n, p = data.shape
    
    # Error Handling: Check if b is less than n
    if b >= n:
        raise ValueError("b should be less than the number of samples n.")
    
    # Error Handling: Check if Q is smaller or equal to the number of possible sub-samples
    if Q > math.comb(n, b):
        raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

    # Generate all possible sub-samples without replacement
    all_subs = list(combinations(range(n), b))
    np.random.seed(42)
    selected_subs = np.random.choice(len(all_subs), min(Q, len(all_subs)), replace=False)
    
    edge_counts_all = np.zeros((p, p, len(lambda_range)))  # Initialize edge count matrix across lambdas and sub-samples

    params_list = list(product(
        selected_subs,
        lambda_range,
        all_subs * len(selected_subs) * len(lambda_range),
        data * len(selected_subs) * len(lambda_range),
        p * len(selected_subs) * len(lambda_range),
        prior_matrix * len(selected_subs) * len(lambda_range)))

    # Use a process pool executor to parallelize the optimization and counting task
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(optimize_for_q_and_j, params_list))

    # Update edge_counts_all with the results
    for q, lambdax, edge_counts in results:
        l = np.where(lambda_range == lambdax)[0][0]  # Find the index of lambdax in lambda_range
        edge_counts_all[:,:,l] += edge_counts

    return edge_counts_all


def objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix):
    """
    Objective function for the piGGM optimization problem.
    Parameters
    ----------
    precision_vector : array-like, shape (p, p)
        The precision vector to be optimised (parameter vector).
    S : array-like, shape (p, p)
        The empirical covariance matrix.
    lambda_np : float
        The regularization parameter for the non-prior edges.
    lambda_wp : float
        The regularization parameter for the prior edges.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.

    Returns
    -------
    objective_value : float
        The objective function value.
    """
    # Reshape precision_vector into a matrix
    p = int(np.sqrt(len(precision_vector)))
    precision_matrix = precision_vector.reshape((p, p))

    if np.isclose(np.linalg.det(precision_matrix), 0):
        raise ValueError("Precision matrix is not invertible.")

    ###### Compute the base objective function ######
    log_det = np.log(np.linalg.det(precision_matrix))
    trace_term = np.trace(np.dot(S, precision_matrix))
    base_objective = -log_det + trace_term
    
    # Identify the entries corresponding to non-zero and zero entries in the prior matrix (Boolean Mask)
    prior_entries = prior_matrix != 0
    non_prior_entries = prior_matrix == 0

    # Compute the separate penalties
    penalty_wp = lambda_wp * np.sum(np.abs(precision_matrix[prior_entries]))
    penalty_np = lambda_np * np.sum(np.abs(precision_matrix[non_prior_entries]))
    
    objective_value = base_objective + penalty_wp + penalty_np
    
    return objective_value

# def subsample_optimiser(data, b, Q, lambda_range, prior_matrix):
#     n, p = data.shape

#     # Error Handling: Check if b is less than n
#     if b >= n:
#         raise ValueError("b should be less than the number of samples n.")
    
#     # Error Handling: Check if Q is smaller or equal to the number of possible sub-samples
#     if Q > math.comb(n, b):
#         raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

#     # Generate all possible sub-samples without replacement
#     all_subs = list(combinations(range(n), b))
#     np.random.seed(42)
#     selected_subs = np.random.choice(len(all_subs), min(Q, len(all_subs)), replace=False)
    
#     edge_counts_all = np.zeros((p, p, len(lambda_range)))  # Initialize edge count matrix across lambdas and sub-samples
#     # Loop for calculating graph structures across lambdas and sub-samples
#     for l, lambdax in enumerate(lambda_range):
#         edge_counts = np.zeros((p, p))  # Initialize edge count matrix for a given lambda
        
#         for q in selected_subs:
#             sub_sample = data[np.array(all_subs[q]), :]
#             S = empirical_covariance(sub_sample)
#             # Optimize the objective function with fixed lambda_wp (e.g., 0.1)
#             initial_precision_vector = np.eye(p).flatten()
#             result = minimize(
#                 subsample_optimiser,
#                 initial_precision_vector,
#                 args=(S, lambdax, lambdax, prior_matrix),
#                 method='L-BFGS-B',
#             )
#             if result.success:
#                 opt_precision_mat = result.x.reshape((p, p))
#                 # Update edge count matrix
#                 edge_counts += (np.abs(opt_precision_mat) > 1e-5).astype(int)  # Assume edge exists if absolute value > 1e-5
            
#         edge_counts_all[:,:,l] += edge_counts
    
#     return edge_counts_all

def estimate_lambda_np(edge_counts_all, Q, lambda_range):
    # Get the dimensions from edge_counts_all
    p, _, _ = edge_counts_all.shape
    J = len(lambda_range)

    # Precompute the N_k_matrix and p_k_matrix, as they do not depend on lambda
    N_k_matrix = np.sum(edge_counts_all, axis=2)
    p_k_matrix = N_k_matrix / (Q * J)

    # Compute theta_lj_matrix, f_k_lj_matrix, and g_l_matrix for all lambdas simultaneously
    theta_matrix = comb(Q, edge_counts_all) * (p_k_matrix[:, :, None] ** edge_counts_all) * ((1 - p_k_matrix[:, :, None]) ** (Q - edge_counts_all))
    f_k_lj_matrix = edge_counts_all / Q
    g_matrix = 4 * f_k_lj_matrix * (1 - f_k_lj_matrix)

    # Reshape the matrices for vectorized operations
    theta_matrix_reshaped = theta_matrix.reshape(J, -1)
    g_matrix_reshaped = g_matrix.reshape(J, -1)

    # Compute the score for each lambda using vectorized operations
    scores = np.sum(theta_matrix_reshaped * (1 - g_matrix_reshaped), axis=1)

    # Find the lambda that maximizes the score
    lambda_np = lambda_range[np.argmax(scores)]
    
    return lambda_np, p_k_matrix, theta_matrix



def estimate_lambda_wp(data, b, Q, p_k_matrix, zks, lambda_range, prior_matrix):
    n, p = data.shape
    results = []

    # reshape the prior matrix to only contain the edges in the lower triangle of the matrix
    wp_tr = [(i, j) for i, j in combinations(range(p), 2) if prior_matrix[i, j] != 0] # THIS SETS THE INDICES FOR ALL VECTORIZED OPERATIONS
    wp_tr_weights = [prior_matrix[comb[0], comb[1]] for comb in wp_tr]
    psis = wp_tr_weights * Q # expansion: add a third dimension of length r, corresponding to the number of prior sources

    p_k_vec = [p_k_matrix[comb[0], comb[1]] for comb in wp_tr]    
    count_mat = np.zeros((len(lambda_range), len(wp_tr))) # Stores zks for each edge across lambdas (shape: lambdas x edges)
    for l in range(len(lambda_range)):
        count_mat[l,:] =  [zks[comb[0], comb[1], l] for comb in wp_tr]

    # Alternative code
    # wp_tr_rows, wp_tr_cols = zip(*wp_tr)  # Unzip the wp_tr tuples into two separate lists
    # z_mat = zks[wp_tr_rows, wp_tr_cols, np.arange(len(lambda_range))[:, None]]


    # calculate mus, vars and tau_tr (=SD of the prior distribution)
    mus = [p_k * Q for p_k in p_k_vec]
    vars = [p_k * (1 - p_k) * Q for p_k in p_k_vec]
    tau_tr = np.sum(np.abs(mus - psis)) / len(wp_tr) # NOTE: alternatively, divide by np.sum(np.abs(wp_tr))

    ######## POSTERIOR DISTRIBUTION ######################################################################
    mus = np.array(mus)
    vars = np.array(vars)
    psis = np.array(psis)

    # Vectorized computation of post_mu and post_var
    post_mu = (mus * tau_tr**2 + psis * vars) / (vars + tau_tr**2)
    post_var = (vars * tau_tr**2) / (vars + tau_tr**2)

    # Since the normal distribution parameters are arrays...
    # Compute the CDF values directly using the formula for the normal distribution CDF
    epsilon = 1e-5
    z_scores_plus = (count_mat + epsilon - post_mu[None, :]) / np.sqrt(post_var)[None, :]
    z_scores_minus = (count_mat - epsilon - post_mu[None, :]) / np.sqrt(post_var)[None, :]
    
    # Compute CDF values using the error function
    # By subtracting 2 values of the CDF, the 1s cancel 
    thetas = 0.5 * (scipy.special.erf(z_scores_plus / np.sqrt(2)) - scipy.special.erf(z_scores_minus / np.sqrt(2)))

    ######### SCORING #####################################################################
    # Frequency, instability, and score
    freq_mat = count_mat / Q                                       # shape: lambdas x edges
    g_mat = 4 * freq_mat * (1 - freq_mat)

    # Scoring function
    scores = np.sum(thetas * (1 - g_mat), axis=1)

    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]

    return lambda_wp, tau_tr, mus


def tau_permutations(data, tau_tr, prior_matrix, wp_tr, Q, mus, N_permutations=10000):
    n, p = data.shape
    # Generate empirical null distribution of tau, similar to GSEA
    tau_perm = np.zeros((N_permutations))

    num_edges = len(wp_tr)

    for _ in range(N_permutations):
        permuted_edges = [tuple(random.sample(range(n), 2)) for _ in range(num_edges)]
        permuted_weights = [prior_matrix[comb[0], comb[1]] for comb in permuted_edges]

        psi_perm = permuted_weights * Q

        tau_perm[_] = np.sum(np.abs(mus - psi_perm)) / num_edges
    
    tau_mean = np.mean(tau_perm)
    tau_perm_normalized = tau_perm / tau_mean
    tau_normalized = tau_tr / tau_mean

    # calculate percentage of taus that are greater than tau_tr, and subtract from 1 to get p-value
    p_value = 1 - (np.sum(tau_perm_normalized >= tau_normalized) / N_permutations)

    return p_value


# Unit Test
def test():
    data = np.random.rand(50, 10)
    b = 20
    Q = 3
    lambda_range = np.linspace(0.01, 1, 2)
    prior_matrix = np.random.rand(10, 10)
    
    # Test subsampler function
    edge_counts_all = subsample_optimiser(data, b, Q, lambda_range, prior_matrix)
    assert edge_counts_all.shape == (10, 10, 10), f"Expected (10, 10, 10), but got {edge_counts_all.shape}"
    
    # Test estimate_lambda_np function
    lambda_np, p_k_matrix, zks = estimate_lambda_np(data, b, Q, lambda_range, edge_counts_all)
    assert lambda_np in lambda_range, f"lambda_np not in lambda_range"
    assert p_k_matrix.shape == (10, 10), f"Expected (10, 10), but got {p_k_matrix.shape}"
    assert zks.shape == (10, 10, 10), f"Expected (10, 10, 10), but got {zks.shape}"
    
    # Test estimate_lambda_wp function
    lambda_wp, tau_tr, mus = estimate_lambda_wp(data, b, Q, p_k_matrix, zks, lambda_range, prior_matrix)
    assert lambda_wp in lambda_range, f"lambda_wp not in lambda_range"
    
    # Test tau_permutations function
    p_value = tau_permutations(data, tau_tr, prior_matrix, [(i, j) for i, j in combinations(range(10), 2)], Q, mus, N_permutations=100)
    assert 0 <= p_value <= 1, f"p_value out of range: {p_value}"
    
    print("All tests passed.")

if __name__ == "__main__":
    test()

