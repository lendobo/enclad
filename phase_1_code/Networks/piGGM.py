import numpy as np
import networkx as nx
import math
from random import sample
from numpy.random import multivariate_normal
from scipy.special import comb, erf
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.linalg import block_diag, eigh, inv
from itertools import combinations
from itertools import product
from sklearn.covariance import empirical_covariance
from concurrent.futures import ProcessPoolExecutor

class SubsampleOptimizer:
    """
    Class for parallel optimisation of the piGGM objective function, across Q sub-samples and J lambdas.

    Attributes
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.
    p : int
        The number of variables.

    Methods
    -------
    objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix)
        The objective function for the piGGM optimization problem.

    optimize_for_q_and_j(params)
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        
    subsample_optimiser(b, Q, lambda_range)
        Optimizes the objective function for all sub-samples and lambda values, using optimize_for_q_and_j.
    """
    def __init__(self, data, prior_matrix):
        self.data = data
        self.prior_matrix = prior_matrix
        self.p = data.shape[1]
        self.selected_sub_idx = None
    

    def objective(self, L_vector, S, lambda_np, lambda_wp, prior_matrix):
        """
        Objective function for the piGGM optimization problem.
        Parameters
        ----------
        L_vector : array-like, shape (p, p)
            The vector of lower diagonal precision matrix to be optimised (parameter vector).
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
        p = self.p

        # Cholesky: Reconstruct the lower triangular matrix L from the vector L_vector
        L = np.zeros((p, p))
        L[np.tril_indices(p)] = L_vector
        # Reconstruct the precision matrix P = LL^T
        precision_matrix = np.dot(L, L.T)

        det_value = np.linalg.det(precision_matrix)

        if det_value <= 0 or np.isclose(det_value, 0):
            print("Warning: non-invertible matrix")
            return np.inf  # return a high cost for non-invertible matrix

        
        # Terms of the base objective function (log-likelihood)
        log_det = np.log(np.linalg.det(precision_matrix))
        trace_term = np.trace(np.dot(S, precision_matrix))
        base_objective = -log_det + trace_term

        # penalty terms
        prior_entries = prior_matrix != 0
        non_prior_entries = prior_matrix == 0
        penalty_wp = lambda_wp * np.sum(np.abs(precision_matrix[prior_entries]))
        penalty_np = lambda_np * np.sum(np.abs(precision_matrix[non_prior_entries]))

        objective_value = base_objective + penalty_wp + penalty_np
        
        return objective_value

    def optimize_for_q_and_j(self, params):
        """
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        Parameters
        ----------
        params : tuple
            Tuple containing the sub-sample index and the lambda value.

        Returns
        -------
        selected_sub_idx : array-like, shape (b)
            The indices of the sub-sample.
        lambdax : float
            The lambda value.
        edge_counts : array-like, shape (p, p)
            The edge counts of the optimized precision matrix.
        """
        selected_sub_idx, lambdax = params
        data = self.data
        p = self.p
        prior_matrix = self.prior_matrix
        sub_sample = data[np.array(selected_sub_idx), :]
        S = empirical_covariance(sub_sample)

        # calculate inverse of S
        try:
            S_inv = inv(S)
        except np.linalg.LinAlgError:
            # print("Warning: non-invertible matrix")
            return selected_sub_idx, lambdax, np.zeros((p, p))

        det_value = np.linalg.det(S_inv)

        # Compute the Cholesky decomposition of the inverse of the empirical covariance matrix
        L_init = np.linalg.cholesky(inv(S))
        # Convert L_init to a vector representing its unique elements
        initial_L_vector = L_init[np.tril_indices(p)]

        result = minimize(
            self.objective,  
            initial_L_vector,
            args=(S, lambdax, lambdax, prior_matrix),
            method='L-BFGS-B',
        )
        if result.success:
            # print(result)
            # Convert result.x back to a lower triangular matrix
            L_opt = np.zeros((p, p))
            L_opt[np.tril_indices(p)] = result.x
            # Compute the optimized precision matrix
            opt_precision_mat = np.dot(L_opt, L_opt.T)
            edge_counts = (np.abs(opt_precision_mat) > 1e-5).astype(int)
            return selected_sub_idx, lambdax, edge_counts
        else:
            return selected_sub_idx, lambdax, np.zeros((p, p))

    def subsample_optimiser(self, b, Q, lambda_range):
        """
        Optimizes the objective function for all sub-samples and lambda values.
        Parameters
        ----------
        b : int
            The size of the sub-samples.
        Q : int
            The number of sub-samples.
        lambda_range : array-like, shape (J)
            The range of lambda values.

        Returns
        -------
        edge_counts_all : array-like, shape (p, p, J)
            The edge counts of the optimized precision matrix for all lambdas.
        """
        n, p = self.data.shape 

        # Error handling: check if b and Q are valid 
        if b >= n:
            raise ValueError("b should be less than the number of samples n.")
        if Q > comb(n, b, exact=True):
            raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

        # Sub-sampling n without replacement
        np.random.seed(42)
        generated_combinations = set()

        while len(generated_combinations) < Q:
            # Generate a random combination
            new_comb = tuple(sorted(sample(range(n), b)))
            # Add the new combination to the set if it's not already present
            generated_combinations.add(new_comb)
        
        # Convert the set of combinations to a list
        self.selected_sub_idx = list(generated_combinations)
        
        # Inferring edges via data-driven optimisation, with prior incorporation
        edge_counts_all = np.zeros((p, p, len(lambda_range)))

        params_list = [(q, lambdax) for q, lambdax in product(self.selected_sub_idx, lambda_range)]

        # Feeding parameters to parallel processing
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.optimize_for_q_and_j, params_list))
        for q, lambdax, edge_counts in results:
            l = np.where(lambda_range == lambdax)[0][0]
            edge_counts_all[:, :, l] += edge_counts

        return edge_counts_all


def estimate_lambda_np(edge_counts_all, Q, lambda_range):
    """
    Estimates the lambda value for the non-prior edges.
    Parameters
    ----------
    edge_counts_all : array-like, shape (p, p, J)
        The edge counts of the optimized precision matrix for all lambdas.
    Q : int
        The number of sub-samples.
    lambda_range : array-like, shape (J)
        The range of lambda values.

    Returns
    -------
    lambda_np : float
        The lambda value for the non-prior edges.
    p_k_matrix : array-like, shape (p, p)
        The probability of an edge being present for each edge, calculated across all sub-samples and lambdas.
    theta_matrix : array-like, shape (p, p, J)
        The probability of z_k edges being present, given a certain lambda.
    """
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



def estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix):
    """
    Estimates the lambda value for the prior edges.
    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    b : int
        The size of the sub-samples.
    Q : int
        The number of sub-samples.
    p_k_matrix : array-like, shape (p, p)
        The probability of an edge being present for each edge, calculated across all sub-samples and lambdas.
    edge_counts_all : array-like, shape (p, p, J)
        The edge counts across sub-samples, for a  a certain lambda.
    lambda_range : array-like, shape (J)
        The range of lambda values.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.

    Returns
    -------
    lambda_wp : float
        The lambda value for the prior edges.
    tau_tr : float
        The standard deviation of the prior distribution.
    mus : array-like, shape (p, p)
        The mean of the prior distribution.
    """
    n, p = data.shape
    results = []

    # reshape the prior matrix to only contain the edges in the lower triangle of the matrix
    wp_tr = [(i, j) for i, j in combinations(range(p), 2) if prior_matrix[i, j] != 0] # THIS SETS THE INDICES FOR ALL VECTORIZED OPERATIONS
    wp_tr_weights = np.array([prior_matrix[comb[0], comb[1]] for comb in wp_tr])
    psis = wp_tr_weights * Q                   # expansion: add a third dimension of length r, corresponding to the number of prior sources

    p_k_vec = [p_k_matrix[comb[0], comb[1]] for comb in wp_tr]    

    count_mat = np.zeros((len(lambda_range), len(wp_tr))) # Stores counts for each edge across lambdas (shape: lambdas x edges)
    for l in range(len(lambda_range)):
        count_mat[l,:] =  [edge_counts_all[comb[0], comb[1], l] for comb in wp_tr]

    # Alternative code for count_mat (=z_mat)
    # wp_tr_rows, wp_tr_cols = zip(*wp_tr)  # Unzip the wp_tr tuples into two separate lists
    # z_mat = zks[wp_tr_rows, wp_tr_cols, np.arange(len(lambda_range))[:, None]]


    # calculate mus, vars and tau_tr (=SD of the prior distribution)
    mus = np.array([p_k * Q for p_k in p_k_vec])
    variances = np.array([p_k * (1 - p_k) * Q for p_k in p_k_vec])
    tau_tr = np.sum(np.abs(mus - psis)) / len(wp_tr) # NOTE: alternatively, divide by np.sum(np.abs(wp_tr))


    ######## POSTERIOR DISTRIBUTION ######################################################################
    mus = np.array(mus)
    variances = np.array(variances)
    psis = np.array(psis)

    # Vectorized computation of post_mu and post_var
    post_mu = (mus * tau_tr**2 + psis * variances) / (variances + tau_tr**2)
    post_var = (variances * tau_tr**2) / (variances + tau_tr**2)

    # Since the normal distribution parameters are arrays...
    # Compute the CDF values directly using the formula for the normal distribution CDF
    epsilon = 1e-5
    z_scores_plus = (count_mat + epsilon - post_mu[None, :]) / np.sqrt(post_var)[None, :]
    z_scores_minus = (count_mat - epsilon - post_mu[None, :]) / np.sqrt(post_var)[None, :]
    
    # Compute CDF values using the error function
    # By subtracting 2 values of the CDF, the 1s cancel 
    thetas = 0.5 * (erf(z_scores_plus / np.sqrt(2)) - erf(z_scores_minus / np.sqrt(2)))

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
    """
    Calculates the p-value for the tau statistic.
    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    tau_tr : float
        The standard deviation of the prior distribution.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.
    wp_tr : array-like, shape (r, 2)
        The indices of the prior edges.
    Q : int
        The number of sub-samples.
    mus : array-like, shape (p, p)
        The mean of the prior distribution.
    N_permutations : int, optional
        The number of permutations to perform. The default is 10000.

    Returns
    -------
    p_value : float
        The p-value for the tau statistic.
    """
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


def test():
    """
    Tests the functions in this file.
    """
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


def synthetic_run(p = 10, n = 50, b = 25, Q = 10, lambda_range = np.linspace(0.01, 0.2, 10)):
    # Set random seed for reproducibility
    np.random.seed(42)

    # TRUE NETWORK
    G = nx.barabasi_albert_graph(p, 5)
    adj_matrix = nx.to_numpy_array(G)
    min_eig = np.min(np.real(eigh(adj_matrix)[0]))
    if min_eig < 0:
        adj_matrix -= 10 * min_eig * np.eye(adj_matrix.shape[0])
    eigenvalues, _ = eigh(adj_matrix)
    print(f'matrix is positive definite: {np.all(eigenvalues > 0)}')
    covariance_matrix = inv(adj_matrix)

    # PRIOR MATRIX
    prior_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            if adj_matrix[i, j] == 1:
                prior_matrix[i, j] = 1.0
                prior_matrix[j, i] = 1.0
    np.fill_diagonal(prior_matrix, 0)

    # DATA MATRIX
    data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_matrix, size=n)

    # MODEL RUN
    optimizer = SubsampleOptimizer(data, prior_matrix)
    edge_counts_all = optimizer.subsample_optimiser(b, Q, lambda_range)

    lambda_np, p_k_matrix, theta_matrix = estimate_lambda_np(edge_counts_all, Q, lambda_range)
    print(lambda_np)

    lambda_wp, tau_tr, mus = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix)
    print(lambda_wp)
    
    return lambda_np, lambda_wp

# # Example usage
# num_runs = 10
# lambda_np_values = []
# lambda_wp_values = []

# for _ in range(num_runs):
#     lambda_np, lambda_wp = synthetic_run()
#     lambda_np_values.append(lambda_np)
#     lambda_wp_values.append(lambda_wp)

# mean_lambda_np = np.mean(lambda_np_values)
# mean_lambda_wp = np.mean(lambda_wp_values)

# print("Mean lambda_np:", mean_lambda_np)
# print("Mean lambda_wp:", mean_lambda_wp)


# Demonstrating the prior penalty estimation
def generate_data_from_network(p=25):
    """
    Generate a data matrix from a synthetic network.
    Returns the adjacency matrix (representing the true network) and the data matrix.
    """
    # Generate a Barabasi-Albert graph
    G = nx.barabasi_albert_graph(p, 3)
    adj_matrix = nx.to_numpy_array(G)

    # Ensure the matrix is positive definite
    min_eig = np.min(np.real(eigh(adj_matrix)[0]))
    if min_eig < 0:
        adj_matrix -= 10 * min_eig * np.eye(adj_matrix.shape[0])

    # Derive the covariance matrix from the adjacency matrix and generate data
    covariance_matrix = inv(adj_matrix)
    data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_matrix, size=50)

    return adj_matrix, data

def modify_prior_matrix(true_prior, random_prior, num_replacements):
    """
    Modify the true prior matrix by replacing specified number of fields with the random prior matrix.
    """
    p = true_prior.shape[0]
    modified_prior = true_prior.copy()

    # Get the indices of the fields to be replaced
    indices = np.array([(i, j) for i in range(p) for j in range(p)])
    selected_indices = indices[np.random.choice(indices.shape[0], num_replacements, replace=False)]

    # Replace the fields in the true prior matrix with the fields from the random prior matrix
    for idx in selected_indices:
        modified_prior[idx[0], idx[1]] = random_prior[idx[0], idx[1]]
        modified_prior[idx[1], idx[0]] = random_prior[idx[1], idx[0]]  # Keep the matrix symmetric

    return modified_prior


# PARAMS
p = 25
n = 50
b = 25
Q = 10
lambda_range = np.linspace(0.01, 0.2, 5)

# Generate a single data matrix from a synthetic network
adj_matrix, data = generate_data_from_network(p)

# Generate the true prior matrix and a random matrix
true_prior = adj_matrix.copy()
np.fill_diagonal(true_prior, 0)
random_prior = np.random.randint(2, size=(25, 25))
np.fill_diagonal(random_prior, 0)

# Generate a sequence of modified prior matrices with increasing number of replacements
replacements = [250, 100, 50, 10, 1]
modified_priors = [] # [modify_prior_matrix(true_prior, random_prior, r) for r in replacements]

modified_priors.append(true_prior)  # Display the first modified prior matrix for reference

# Run optimization
optimizer = SubsampleOptimizer(data, true_prior)
edge_counts_all = optimizer.subsample_optimiser(b, Q, lambda_range)

# Estimate lambda_wp for each prior matrix
lambda_wp_values = []
for prior_matrix in modified_priors:
    # lambda_np, p_k_matrix, theta_matrix = estimate_lambda_np(edge_counts_all, 10, np.linspace(0.01, 0.2, 10))
    lambda_wp, _, _ = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix)
    lambda_wp_values.append(lambda_wp)

print(lambda_wp_values)



####### TODO #######
# SWITCH TO PSEUDO LOG-LIKELIHOOD
# (mostly DONE) Vectorize the for loops 
# (DONE) implement parallelization for Q * J optimization runs 
# Once it runs, implement alternative and check if results are still the same
# check if the denominator of eq (12) is implemented correctly (Run both versions and compare)

###### Possible adjustments to experiment with #######
# Currently, we assume an edge is present if the optimized precision matrix has an absolute value > 1e-5.
# If the resulting network is too dense, we might have to change this threshold. However, the LASSO pushes many values to exactly 0, 
# so this might not be a problem.

