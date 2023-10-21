import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from random import sample
from numpy.random import multivariate_normal
from scipy.special import comb, erf
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.linalg import block_diag, eigh, inv
from itertools import combinations
from itertools import product
from sklearn.covariance import empirical_covariance, GraphicalLasso
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from concurrent.futures import as_completed
import sys
import pickle

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

        logdet_sign, logdet_value = np.linalg.slogdet(precision_matrix)

        if logdet_sign == -1 or np.isclose(logdet_value, 0):
            print("Optimizer: non-invertible matrix")
            return np.inf  # return a high cost for non-invertible matrix

        
        # Terms of the base objective function (log-likelihood)
        log_det = np.linalg.slogdet(precision_matrix)[1]
        trace_term = np.trace(np.dot(S, precision_matrix))
        base_objective = -log_det + trace_term

        # # penalty terms
        # prior_entries = prior_matrix != 0
        # non_prior_entries = prior_matrix == 0
        # penalty_wp = lambda_wp * np.sum(np.abs(precision_matrix[prior_entries]))
        # penalty_np = lambda_np * np.sum(np.abs(precision_matrix[non_prior_entries]))
        penalty = lambda_wp * np.sum(np.abs(precision_matrix))

        objective_value = base_objective + penalty # penalty_wp + penalty_np
        
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
            return selected_sub_idx, lambdax, edge_counts, 1
        else:
            # print("Optimization did not succeed.")
            # print("Reason:", result.message)
            # print("Status code:", result.status)
            # print("Objective function value:", result.fun)
            # print("Number of function evaluations:", result.nfev)
            # print("Number of iterations:", result.nit)
            return selected_sub_idx, lambdax, np.zeros((p, p)), 0

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

        results = []

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(params_list), desc="Processing", ncols=100)

        # Feeding parameters to parallel processing
        with ProcessPoolExecutor() as executor:
            # Using executor.submit to get futures and as_completed to retrieve results in the order they complete
            futures = {executor.submit(self.optimize_for_q_and_j, param): param for param in params_list}
            for future in as_completed(futures):
                # Update the progress bar
                progress_bar.update(1)
                results.append(future.result())

        progress_bar.close()

        success_counts = np.zeros(len(lambda_range))

        for q, lambdax, edge_counts, success_check in results:
            l = np.where(lambda_range == lambdax)[0][0]
            edge_counts_all[:, :, l] += edge_counts
            success_counts[l] += success_check
        success_perc = success_counts / Q

        return edge_counts_all, success_counts, success_perc












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

    # reshape the prior matrix to only contain the edges in the lower triangle of the matrix
    wp_tr_idx = [(i, j) for i, j in combinations(range(p), 2) if prior_matrix[i, j] != 0] # THIS SETS THE INDICES FOR ALL VECTORIZED OPERATIONS
    
    # wp_tr_weights and p_k_vec give the prob of an edge in the prior and the data, respectively
    wp_tr_weights = np.array([prior_matrix[ind[0], ind[1]] for ind in wp_tr_idx])
    p_k_vec = np.array([p_k_matrix[ind[0], ind[1]] for ind in wp_tr_idx])

    count_mat = np.zeros((len(lambda_range), len(wp_tr_idx))) # Stores counts for each edge across lambdas (shape: lambdas x edges)
    for l in range(len(lambda_range)):
        count_mat[l,:] =  [edge_counts_all[ind[0], ind[1], l] for ind in wp_tr_idx]

    # Alternative code for count_mat (=z_mat)
    # wp_tr_rows, wp_tr_cols = zip(*wp_tr)  # Unzip the wp_tr tuples into two separate lists
    # z_mat = zks[wp_tr_rows, wp_tr_cols, np.arange(len(lambda_range))[:, None]]


    ######### DATA DISTRIBUTION #####################################################################
    # calculate mus, vars 
    mus = p_k_vec * Q
    variances = p_k_vec * (1 - p_k_vec) * Q

    ######### PRIOR DISTRIBUTION #####################################################################
    #psi (=prior mean)
    psis = wp_tr_weights * Q                   # expansion to multipe prior sources: add a third dimension of length r, corresponding to the number of prior sources
    # tau_tr (=SD of the prior distribution)
    tau_tr = np.sum(np.abs(mus - psis)) / len(wp_tr_idx) # NOTE: eq. 12 alternatively, divide by np.sum(np.abs(wp_tr))


    ######## POSTERIOR DISTRIBUTION ######################################################################
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
    # print('shape of thetas: ', {thetas.shape})

    ######### SCORING #####################################################################
    # Frequency, instability, and score
    freq_mat = count_mat / Q                                       # shape: lambdas x edges
    g_mat = 4 * freq_mat * (1 - freq_mat)

    # Scoring function
    scores = np.sum(thetas * (1 - g_mat), axis=1)

    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]

    # print(f'mus; {mus}')
    # print('variances: ', variances)
    # print(f'psis: {psis}')
    # print(f'tau_tr: {tau_tr}')
    # print(f'post_mu: {post_mu}')
    # print(f'post_var: {post_var}')

    # Writing to a file for diagnosis
    original_stdout = sys.stdout
    with open(f'out/old_diagnosis.txt', 'w') as f:
        sys.stdout = f
        for e in range(len(wp_tr_idx)):
            print('\n\n\n')
            for j in range(len(lambda_range)):
                print('\n')
                print(f'lambda: {lambda_range[j]}')
                print(f'z_k_{str(e)}: {count_mat[j, e]}')
                print(f'prior_mu: {psis[e]}')
                print(f'data_mu: {mus[e]}')
                print(f'posterior_mu: {post_mu[e]}')
                print('posterior var: ', post_var[e])
                print(f'thetas: {thetas[j, e]}')
                print(f'g_mat: {g_mat[j, e]}')
                print(f'scores: {scores[j]}')
    
    sys.stdout = original_stdout

    return lambda_wp, tau_tr, mus












    # return lambda_np, lambda_wp, edge_counts_all

def optimize_graph(data, prior_matrix, lambda_np, lambda_wp):
    """
    Optimizes the objective function using the entire data set and the estimated lambdas.

    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix.
    lambda_np : float
        The regularization parameter for the non-prior edges.
    lambda_wp : float
        The regularization parameter for the prior edges.

    Returns
    -------
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    """
    n, p = data.shape
    optimizer = SubsampleOptimizer(data, prior_matrix)
    S = empirical_covariance(data)
    Eye = np.eye(p)

    # Compute the Cholesky decomposition of the inverse of the empirical covariance matrix
    try:
        epsilon = 1e-3
        L_init = np.linalg.cholesky(inv(Eye + epsilon * np.eye(p)))
    except np.linalg.LinAlgError:
        print("Initial Guess: non-invertible matrix")
        return np.zeros((p, p))
    
    # Convert L_init to a vector representing its unique elements
    initial_L_vector = L_init[np.tril_indices(p)]

    result = minimize(
        optimizer.objective,  
        initial_L_vector,
        args=(S, lambda_np, lambda_wp, prior_matrix),
        method='L-BFGS-B',
    )

    if result.success:
        # Convert result.x back to a lower triangular matrix
        L_opt = np.zeros((p, p))
        L_opt[np.tril_indices(p)] = result.x
        # Compute the optimized precision matrix
        opt_precision_mat = np.dot(L_opt, L_opt.T)
        return opt_precision_mat
    else:
        return np.zeros((p, p))



def synthetic_run(lambda_np, lambda_wp, p = 10, n = 500, b = 250, Q = 50, lambda_range = np.linspace(0.01, 0.05, 20)):
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

    # # MODEL RUN
    optimizer = SubsampleOptimizer(data, prior_matrix)
    edge_counts_all, success_counts, success_perc = optimizer.subsample_optimiser(b, Q, lambda_range)
    # print(f'check0: edge_counts_all: {edge_counts_all}')
    # lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)
    # print(f'check1: lambda_np: {lambda_np}')
    # # Estimate true p lambda_wp
    # lambda_wp, _, _ = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix)
    # print(f'check2: lambda_wp: {lambda_wp}')
    
    opt_precision_mat = np.zeros((p,p)) # optimize_graph(data, prior_matrix, lambda_np, lambda_wp)
    # print(f'check3: opt_precision_mat: {opt_precision_mat}')

    return lambda_np, lambda_wp, opt_precision_mat, adj_matrix, edge_counts_all, success_counts, success_perc

p = 100
n = 80
b = int(n / 2)
Q = 70
lambda_granularity = 20
lambda_range_np = np.linspace(0.01, 0.4, lambda_granularity)
lambda_range_wp = np.linspace(0.01, 0.4, lambda_granularity)

lambda_np, lambda_wp, opt_precision_mat, adj_matrix, edge_counts_all, success_counts, success_perc = synthetic_run(0,0, p, n, b, Q, lambda_range_np)

# # save edge_coutns_all to pkl file
# with open(f'out/edge_counts_all.pkl{p,n,b,Q,len(lambda_range_np)}', 'wb') as f:
#     pickle.dump(edge_counts_all, f)
# # save success_counts to pkl file
# with open(f'out/success_counts.pkl{p,n,b,Q,len(lambda_range_np)}', 'wb') as f:
#     pickle.dump(success_counts, f)

# # open edge_counts_all from pkl file
# with open(f'out/edge_counts_all.pkl{p,n,b,Q,len(lambda_range_np)}', 'rb') as f:
#     edge_counts_all = pickle.load(f)
# # open edge_counts_all from pkl file
# with open(f'out/success_counts.pkl{p,n,b,Q,len(lambda_range_np)}', 'rb') as f:
#     success_counts = pickle.load(f)

print(success_perc)
# get the right dimensions for edge_counts_all for plotting
# mask = np.triu(np.ones((p, p, len(lambda_range_np))), k=1)
# Element-wise multiplication and sum along the first two dimensions
edges_per_lambda = [np.sum(np.triu(edge_counts_all[:,:,i], k=1)) / success_counts[i] if success_counts[i] != 0 else 0 for i in range(len(lambda_range_np))]

# plot the fitted curves to the left and right of the knee point
plt.figure(figsize=(10, 6))
plt.scatter(lambda_range_np, edges_per_lambda)
# plt.plot(penalty_values[:knee_point_index], linear_func(penalty_values[:knee_point_index], *params_left), color="red")
# plt.plot(penalty_values[knee_point_index+1:], linear_func(penalty_values[knee_point_index+1:], *params_right), color="green")
plt.xlabel("Penalty value")
plt.ylabel("Number of edges")
plt.show()

    