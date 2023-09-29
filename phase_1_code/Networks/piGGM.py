import numpy as np
import math
from random import sample
from scipy.special import comb
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.linalg import block_diag
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
    

    def objective(self, precision_vector, S, lambda_np, lambda_wp, prior_matrix):
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
        p = self.p
        precision_matrix = precision_vector.reshape((p, p))
        
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
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # print("Warning: non-invertible matrix")
            return selected_sub_idx, lambdax, np.zeros((p, p))

        det_value = np.linalg.det(S_inv)
        initial_precision_vector = S_inv.flatten() # np.eye(p).flatten()
        result = minimize(
            self.objective,  
            initial_precision_vector,
            args=(S, lambdax, lambdax, prior_matrix),
            method='L-BFGS-B',
        )
        if result.success:
            opt_precision_mat = result.x.reshape((p, p))
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



def estimate_lambda_wp(data, b, Q, p_k_matrix, zks, lambda_range, prior_matrix):
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
    zks : array-like, shape (p, p, J)
        The probability of z_k edges being present, given a certain lambda.
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
    wp_tr_weights = [prior_matrix[comb[0], comb[1]] for comb in wp_tr]
    psis = wp_tr_weights * Q                   # expansion: add a third dimension of length r, corresponding to the number of prior sources

    p_k_vec = [p_k_matrix[comb[0], comb[1]] for comb in wp_tr]    
    count_mat = np.zeros((len(lambda_range), len(wp_tr))) # Stores zks for each edge across lambdas (shape: lambdas x edges)
    for l in range(len(lambda_range)):
        count_mat[l,:] =  [zks[comb[0], comb[1], l] for comb in wp_tr]

    # Alternative code for count_mat (=z_mat)
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


######## Computations ####################################################################################################################
# DATA MATRIX
# Set random seed for reproducibility
np.random.seed(42)

# Dimensions
n, p = 50, 10  # 50 samples, 10 variables

# Create block diagonal covariance matrix
block_size = p // 2  # assuming two blocks for simplicity
block1 = np.random.rand(block_size, block_size)
block2 = np.random.rand(block_size, block_size)
cov_block1 = np.dot(block1, block1.transpose())
cov_block2 = np.dot(block2, block2.transpose())
cov_matrix = block_diag(cov_block1, cov_block2)  # This will create a block diagonal covariance matrix

# Generate synthetic data (log-normal distribution)
mean = np.zeros(p)  # Assuming a mean of 0 for simplicity
log_data = np.random.multivariate_normal(mean, cov_matrix, size=n)

# Exponentiate to emulate log-normal distribution
data = np.exp(log_data)

# Display the first 5 rows of the generated data
data[:5]


# PRIOR MATRIX
# Confidence of prior edges is between 0.7 and 0.9
prior_matrix = np.random.uniform(0.7, 0.9, size=(p, p))

# Setting edges with no prior to 0
sparsity_pattern = np.random.choice([0, 1], size=(p, p), p=[0.8, 0.2])
prior_matrix *= sparsity_pattern


# MODEL PARAMETERS
b = 25
Q = 3
lambda_range = np.linspace(0.4, 0.6, 1)

optimizer = SubsampleOptimizer(data, prior_matrix)
edge_counts_all = optimizer.subsample_optimiser(b, Q, lambda_range)



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

