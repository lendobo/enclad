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
from tqdm import tqdm
import sys

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
            print("Optimizer: non-invertible matrix")
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
        Eye = np.eye(p)
        
        # Compute the Cholesky decomposition of the inverse of the empirical covariance matrix
        try:
            epsilon = 1e-3
            L_init = np.linalg.cholesky(inv(Eye + epsilon * np.eye(p)))
        except np.linalg.LinAlgError:
            print("Initial Guess: non-invertible matrix")
            return selected_sub_idx, lambdax, np.zeros((p, p))
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
        # np.random.seed(42)
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



def estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrices):
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
    prior_matrices : array-like, shape (p, p, r)
        The prior matrices. Separate priors are stacked along axis = 2. Edges may have different / overlapping priors.

    Returns
    -------
    lambda_wp : float
        The lambda value for the prior edges.
    var_KL : float
        The standard deviation of the prior distribution.
    mus : array-like, shape (p, p)
        The mean of the prior distribution.
    """ 
    n, p = data.shape

    prior_taus = np.zeros(prior_matrices.shape[2])

    for pri_idx in range(prior_matrices.shape[2]):
        # indices of the edges that are non-zero for this specific prior
        wp_tr_idx = [(i, j) for i, j in combinations(range(p), 2) if prior_matrices[i, j, pri_idx] != 0] # These are also used to index the weights, p_k_vec, and count_mat
        
        # wp_tr_weights and p_k_vec give the prob of an edge in the prior and the data, respectively
        wp_tr_weights = np.array([prior_matrices[ind[0], ind[1], pri_idx] for ind in wp_tr_idx])

        # DATA DISTRIBUTION, mean values for edges
        p_k_vec = np.array([p_k_matrix[ind[0], ind[1]] for ind in wp_tr_idx])
        mus = p_k_vec * Q                             # mus from data, at indices for edges of this single prior

        # PRIOR DISTRIBUTION, single prior
        # psi (=prior expected value across Q subsamples)
        psis = wp_tr_weights * Q                                      # expansion to multipe prior sources: add a third dimension of length r, corresponding to the number of prior sources
        # tau_tr (=SD of the prior distribution)
        prior_taus[pri_idx] = np.sum(np.abs(mus - psis)) / len(wp_tr_idx) # NOTE: eq. 12 alternatively, divide by np.sum(np.abs(wp_tr))


    # Find the edges that are non-zero in any of the prior matrices
    any_nonzero = np.any(prior_matrices != 0, axis=2)
    wp_tr_across_priors = [(i, j) for i, j in combinations(range(p), 2) if any_nonzero[i, j]]

    # Initialize structures to store Gaussian mixture parameters for each edge
    mus_KL = np.zeros(len(wp_tr_across_priors))                       # Means of the prior distributions, after gaussian mixture approximation via Kullback-Leibler divergence
    vars_KL = np.zeros(len(wp_tr_across_priors))                      # Variances of the prior distributions, after gaussian mixture approximation via Kullback-Leibler divergence

    # Compute weights for the Gaussian mixture
    a_ti = np.sum(prior_taus) / prior_taus                            # reflects the stability of a prior 
    w_ti = a_ti / np.sum(a_ti)                                        # reflects the instability of a prior in relation to the total instability

    for e, edge in enumerate(wp_tr_across_priors):
        i, j = edge

        # Extract the non-zero priors for this edge
        edge_prior_weights = prior_matrices[i,j,:] # prior_matrices[i, j, prior_matrices[i, j] != 0]

        psi_ti_select = edge_prior_weights[edge_prior_weights != 0] * Q             # psis of all priors for that edge
        prior_taus_select = prior_taus[edge_prior_weights != 0]                     # Taus of all priors for that edge
        w_ti_select = w_ti[edge_prior_weights != 0]                                 # weights of all priors for that edge

        mus_KL[e] = np.sum(w_ti_select * psi_ti_select)
        vars_KL[e] = np.sum(w_ti_select * (prior_taus_select**2 + (psi_ti_select - mus_KL[e])**2))

    ######### DATA DISTRIBUTION #####################################################################
    # count_mat stores counts for each edge across lambdas (shape: lambdas x edges)
    count_mat_all_p = np.zeros((len(lambda_range), len(wp_tr_across_priors))) 
    for l in range(len(lambda_range)):
        count_mat_all_p[l,:] =  [edge_counts_all[ind[0], ind[1], l] for ind in wp_tr_across_priors]
    # calculate mus, vars
    p_k_vec_across_priors = np.array([p_k_matrix[ind[0], ind[1]] for ind in wp_tr_across_priors])
    mus_d_all_p = p_k_vec_across_priors * Q                        # mus from data, at indices for edges across priors
    vars_d_all_p = p_k_vec_across_priors * (1 - p_k_vec_across_priors) * Q

    ######## POSTERIOR DISTRIBUTION ######################################################################
    # Vectorized computation of post_mu and post_var
    post_mu = (mus_d_all_p * vars_KL + mus_KL * vars_d_all_p) / (vars_d_all_p + vars_KL) # NOTE: Should the vars_KL still be squared? tau was squared for single prior
    post_var = (vars_d_all_p * vars_KL) / (vars_d_all_p + vars_KL)

    # Since the normal distribution parameters are arrays...
    # Compute the CDF values directly using the formula for the normal distribution CDF
    epsilon = 1e-5
    z_scores_plus = (count_mat_all_p + epsilon - post_mu[None, :]) / np.sqrt(post_var)[None, :]
    z_scores_minus = (count_mat_all_p - epsilon - post_mu[None, :]) / np.sqrt(post_var)[None, :]
    
    # Compute CDF values using the error function
    # By subtracting 2 values of the CDF, the 1s cancel 
    thetas = 0.5 * (erf(z_scores_plus / np.sqrt(2)) - erf(z_scores_minus / np.sqrt(2)))

    ######### SCORING #####################################################################
    # Frequency, instability, and score
    freq_mat = count_mat_all_p / Q                                       # shape: lambdas x edges
    g_mat = 4 * freq_mat * (1 - freq_mat)

    # Scoring function
    scores = np.sum(thetas * (1 - g_mat), axis=1)

    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]
    

    return lambda_wp, prior_taus, mus












def synthetic_run(run_nm, num_priors = 2, p = 50, n = 500, b = 250, Q = 50, lambda_range = np.linspace(0.01, 0.05, 20)):
    # # Set random seed for reproducibility
    # np.random.seed(42)

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


    # Multi-prior Matrix
    prior_matrices = np.zeros((p,p,num_priors))
    booleans = [np.random.rand() < ind / (num_priors + 1) for ind in range(num_priors)]

    for i in range(p):
        for j in range(i, p):
            if adj_matrix[i, j] != 0:
                prior_matrices[i, j, 0] = 1
                prior_matrices[j, i, 0] = 1
    np.fill_diagonal(prior_matrices[:,:,0], 0)

    for p in range (1, num_priors):
        for i in range(p):
            for j in range(i, p):
                if np.random.rand() < booleans[p]:
                    prior_matrices[i, j, p] = (prior_matrices[i, j, p] - 1) * -1
                    prior_matrices[j, i, p] = prior_matrices[i, j, p]
        np.fill_diagonal(prior_matrices[:,:,p], 0)
    # scale
    prior_matrices = prior_matrices * 0.9


    # SYNTHETIC DATA
    data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)


    # MODEL RUN
    prior_matrices_flat = np.sum(prior_matrices, axis=2)
    optimizer = SubsampleOptimizer(data, prior_matrices_flat)
    edge_counts_all = optimizer.subsample_optimiser(b, Q, lambda_range)
    lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)
    # Estimate true p lambda_wp
    lambda_wp, prior_taus, _ = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrices)


    return lambda_np, lambda_wp, prior_taus

# Example usage
num_runs = 10
p = 10
n = 500
b = int(n / 2)
Q = 50
num_priors = 3
lambda_granularity = 10
lambda_range = np.linspace(0.03, 0.2, lambda_granularity)

lambda_np_values = []
lambda_wp_values = []

# Set the filenames based on initial parameters
lambda_wp_filename = f'out/lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'

for _ in tqdm(range(num_runs)):
    lambda_np, lambda_wp, prior_taus = synthetic_run(_, num_priors=num_priors, p=p, n=n, b=b, Q=Q, lambda_range=lambda_range)
    print(f'lambda_np: {lambda_np}, lambda_wp: {lambda_wp}')
    print(f'prior taus: {prior_taus}')

    lambda_np_values.append(lambda_np)
    lambda_wp_values.append(lambda_wp)

    # # save lambda_wp values and random_lambda_wp values to csv, every 10 runs
    # if _ % 10 == 0:
    #     with open(lambda_wp_filename, 'a') as f:
    #         np.savetxt(f, lambda_wp_values, delimiter=',')
        
    #     with open(random_lambda_wp_filename, 'a') as f:
    #         np.savetxt(f, random_lambda_wp_values, delimiter=',')

# RESULTS
mean_lambda_np = np.mean(lambda_np_values)
mean_lambda_wp = np.mean(lambda_wp_values)
mean_random_lambda_wp = np.mean(random_lambda_wp_values)

print("Mean lambda_np:", mean_lambda_np)
print("Mean lambda_wp:", mean_lambda_wp)
print("Mean random lambda_wp:", mean_random_lambda_wp)

# Save the results to csv
mean_lambda_wp_filename = f'out/mean_lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'
mean_random_lambda_wp_filename = f'out/mean_random_lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'
edge_counts_all_runs_filename = f'out/edge_counts_all_runs_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'

with open(mean_lambda_wp_filename, 'a') as f:
    np.savetxt(f, [mean_lambda_wp], delimiter=',')
with open(mean_random_lambda_wp_filename, 'a') as f:
    np.savetxt(f, [mean_random_lambda_wp], delimiter=',')
# Flatten and stack edge_counts_all_runs
stacked_edge_counts_all_runs = np.vstack([arr.flatten() for arr in edge_counts_all_runs])
with open(edge_counts_all_runs_filename, 'a') as f:
    np.savetxt(f, stacked_edge_counts_all_runs, delimiter=',')
# To open and reshape back into original shape:
# loaded_data = np.loadtxt(filename, delimiter=',')
# list_of_reshaped_arrays = [row.reshape((p, p, J)) for row in loaded_data]

# TODO
# what are the implications of penalizing all edges by same lambda, even though some edges might have different 
# priors with different reliability? Shouldn't this be reflected in the penalty of that specific edge?
