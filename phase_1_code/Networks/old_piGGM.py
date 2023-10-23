import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from random import sample
import random
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
import warnings
from sklearn.exceptions import ConvergenceWarning

# from piGGM.py import optimize_graph, evaluate_reconstruction

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

        # Graphical Lasso for precision matrix inference via coordinate descent
        model = GraphicalLasso(alpha=lambdax, mode='cd', max_iter=100)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning)  # Convert UserWarning to errors
            warnings.filterwarnings("error", category=ConvergenceWarning)  # Convert ConvergenceWarning to errors
            try:
                model.fit(sub_sample)
                precision_matrix = model.precision_
                edge_counts = (np.abs(precision_matrix) > 1e-5).astype(int)
                return selected_sub_idx, lambdax, edge_counts, precision_matrix, 1
            except (UserWarning, ConvergenceWarning) as e:
                print(f"Warning caught: {str(e)}")
                return selected_sub_idx, lambdax, np.zeros((p,p)), precision_matrix, 0


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
        # random.seed(42)
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

        for q, lambdax, edge_counts, prec_mat, success_check in results:
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
    
    sys.stdout = original_stdout

    return lambda_wp, tau_tr, mus


def optimize_graph(data, prior_matrix, lambda_val):
    """
    Optimizes the objective function using the entire data set and the estimated lambda.

    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix.
    lambda_val : float
        The regularization parameter for the edges.

    Returns
    -------
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    """
    # Use GraphicalLasso to estimate the precision matrix
    model = GraphicalLasso(alpha=lambda_val, mode='cd', max_iter=100)
    try:
        model.fit(data)
        return model.precision_
    except Exception as e:
        print(f"Optimization did not succeed due to {str(e)}")
        return np.zeros((data.shape[1], data.shape[1]))




def synthetic_run(p = 10, n = 500, b = 250, Q = 50, lambda_range = np.linspace(0.01, 0.05, 20)):
    # # Set random seed for reproducibility
    np.random.seed(42)

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
    lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)
    # # Estimate true p lambda_wp
    # lambda_wp, tau_tr, mus = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix)
    
    # print(lambda_np)
    # lambda_np = 0.078 # 0.07157894736842105

    opt_precision_mat = optimize_graph(data, prior_matrix, lambda_np)

    return opt_precision_mat, adj_matrix, edge_counts_all, success_counts, success_perc, lambda_np


def parallel_synthetic_run(params):
    p, n, b, Q, lambda_range = params
    return synthetic_run(p=p, n=n, b=b, Q=Q, lambda_range=lambda_range)


def synthetic_sweep(p_range=[10], n=60, b_values=[250], Q_values=[50], 
                           lambda_ranges=[np.linspace(0.01, 0.05, 20)]):
    
    # Ensure the input parameters are in list format
    if not isinstance(p_range, list): p_range = [p_range]
    if not isinstance(b_values, list): b_values = [b_values]
    if not isinstance(Q_values, list): Q_values = [Q_values]
    if not isinstance(lambda_ranges, list): lambda_ranges = [lambda_ranges]

    results_dict = {}

    # Create a list of parameter combinations
    param_combinations = list(product(p_range, [n], b_values, Q_values, lambda_ranges))
    
    # Use ProcessPoolExecutor to run in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(parallel_synthetic_run, param_combinations))
    
    for params, result in zip(param_combinations, results):
        key = tuple(list(params[:-1]) + [str(params[-1])])
        results_dict[key] = {
            'opt_precision_mat': result[0],
            'adj_matrix': result[1],
            'edge_counts_all': result[2],
            'success_counts': result[3],
            'success_perc': result[4],
            'lambda_np': result[5]
        }
    
    return results_dict



def evaluate_reconstruction(adj_matrix, opt_precision_mat, threshold=1e-5):
    """
    Evaluate the accuracy of the reconstructed adjacency matrix.

    Parameters
    ----------
    adj_matrix : array-like, shape (p, p)
        The original adjacency matrix.
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    threshold : float, optional
        The threshold for considering an edge in the precision matrix. Default is 1e-5.

    Returns
    -------
    metrics : dict
        Dictionary containing precision, recall, f1_score, and jaccard_similarity.
    """
    # Convert the optimized precision matrix to binary form
    reconstructed_adj = (np.abs(opt_precision_mat) > threshold).astype(int)
    print(reconstructed_adj)
    np.fill_diagonal(reconstructed_adj, 0)

    # True positives, false positives, etc.
    tp = np.sum((reconstructed_adj == 1) & (adj_matrix == 1))
    fp = np.sum((reconstructed_adj == 1) & (adj_matrix == 0))
    fn = np.sum((reconstructed_adj == 0) & (adj_matrix == 1))
    tn = np.sum((reconstructed_adj == 0) & (adj_matrix == 0))

    # Precision, recall, F1 score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Jaccard similarity
    jaccard_similarity = tp / (tp + fp + fn)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'jaccard_similarity': jaccard_similarity
    }

    return metrics

l_lo = 0 # 0.042105263157894736
l_hi = 0.4 # 0.12631578947368421
lambda_range = np.linspace(l_lo, l_hi, 20)

p_range = 10
n = 300
b_values = int((2 / 3) * n)
Q_values = 500


lambda_estimates = []


# Create a single figure and axis for all plots
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(30):
    results_single = synthetic_sweep(p_range=p_range, n = n, b_values=b_values, Q_values=Q_values, lambda_ranges=lambda_range)
    # save to pkl file
    with open(f'phase_1_code/Networks/out/results_single_{p_range,n,b_values,Q_values,len(lambda_range)}.pkl', 'wb') as f:
        pickle.dump(results_single, f)

    # Get the lambda estimate
    lambda_estimate = results_single[(p_range, n, b_values, Q_values, str(lambda_range))]['lambda_np']
    lambda_estimates.append(lambda_estimate)
    print(lambda_estimate)

    edge_counts_all = results_single[(p_range, n, b_values, Q_values, str(lambda_range))]['edge_counts_all']

    success_counts = results_single[(p_range, n, b_values, Q_values, str(lambda_range))]['success_counts']
    success_perc = np.sum(results_single[(p_range, n, b_values, Q_values, str(lambda_range))]['success_perc']) / len(lambda_range)

    # print(success_perc)
    # get the right dimensions for edge_counts_all for plotting
    # mask = np.triu(np.ones((p, p, len(lambda_range))), k=1)
    # Element-wise multiplication and sum along the first two dimensions
    edges_per_lambda = [np.sum(np.triu(edge_counts_all[:,:,i], k=1)) / success_counts[i] if success_counts[i] != 0 else 0 for i in range(len(lambda_range))]

    # Add the scatter plot to the shared figure and provide a label for the legend
    ax.plot(lambda_range, edges_per_lambda, label=f'Lambda estimate: {lambda_estimate}')

    opt_precision_mat = results_single[(p_range, n, b_values, Q_values, str(lambda_range))]['opt_precision_mat']
    adj_matrix = results_single[(p_range, n, b_values, Q_values, str(lambda_range))]['adj_matrix']

    # evaluate metrics  
    metrics = evaluate_reconstruction(adj_matrix, opt_precision_mat)
    print(metrics)



# # Set the axis labels and title
# ax.set_xlabel("Penalty value")
# ax.set_ylabel("Number of edges")
# ax.set_title('Comparison of different runs')

# # Display the legend
# ax.legend()

# # Display the final figure
# plt.show()

# save lambda_estimates to pkl file
with open(f'phase_1_code/Networks/out/lambda_estimates_{p_range,n,b_values,Q_values,len(lambda_range)}.pkl', 'wb') as f:
    pickle.dump(lambda_estimates, f)

# plot a distribution of lambda estimates
plt.figure(figsize=(10, 6))
plt.hist(lambda_estimates)
plt.xlabel("Lambda estimate")
plt.ylabel("Frequency")
plt.title('Distribution of lambda estimates')
plt.show()





# p = 300
# n = 500
# b = int(n / 2)
# Q = 50
# lambda_granularity = 20
# lambda_range_np = np.linspace(0.01, 0.4, lambda_granularity)
# lambda_range_wp = np.linspace(0.01, 0.4, lambda_granularity)

# lambda_np, lambda_wp, opt_precision_mat, adj_matrix, edge_counts_all, success_counts, success_perc = synthetic_run(0,0, p, n, b, Q, lambda_range_np)

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

# print(success_perc)
# # get the right dimensions for edge_counts_all for plotting
# # mask = np.triu(np.ones((p, p, len(lambda_range_np))), k=1)
# # Element-wise multiplication and sum along the first two dimensions
# edges_per_lambda = [np.sum(np.triu(edge_counts_all[:,:,i], k=1)) / success_counts[i] if success_counts[i] != 0 else 0 for i in range(len(lambda_range_np))]

# # plot the fitted curves to the left and right of the knee point
# plt.figure(figsize=(10, 6))
# plt.scatter(lambda_range_np, edges_per_lambda)
# # plt.plot(penalty_values[:knee_point_index], linear_func(penalty_values[:knee_point_index], *params_left), color="red")
# # plt.plot(penalty_values[knee_point_index+1:], linear_func(penalty_values[knee_point_index+1:], *params_right), color="green")
# plt.xlabel("Penalty value")
# plt.ylabel("Number of edges")
# plt.show()

    