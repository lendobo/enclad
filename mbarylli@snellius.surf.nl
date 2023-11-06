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
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from concurrent.futures import ProcessPoolExecutor
from mpi4py import MPI
from tqdm import tqdm
from concurrent.futures import as_completed
import sys
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
import os
import argparse


# Activate the automatic conversion of numpy objects to R objects
numpy2ri.activate()

# Define the R function for weighted graphical lasso
ro.r('''
weighted_glasso <- function(data, penalty_matrix, nobs) {
  library(glasso)
  result <- glasso(s=as.matrix(data), rho=penalty_matrix, nobs=nobs)
  return(list(precision_matrix=result$wi, edge_counts=result$wi != 0))
}
''')

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

        # Number of observations
        nobs = sub_sample.shape[0]

        # Penalty matrix (adapt this to your actual penalty matrix logic)
        penalty_matrix = lambdax * np.ones((p,p)) # prior_matrix

        # print(f'P: {p}')

        # Call the R function from Python
        weighted_glasso = ro.globalenv['weighted_glasso']
        try:
            result = weighted_glasso(S, penalty_matrix, nobs)
            if 'error' in result.names:
                print(f"R Error or Warning: {result.rx('message')[0]}")
                return selected_sub_idx, lambdax, np.zeros((p,p)), np.zeros((p,p)), 0
            else:
                precision_matrix = np.array(result.rx('precision_matrix')[0])
                edge_counts = (np.abs(precision_matrix) > 1e-5).astype(int)
                return selected_sub_idx, lambdax, edge_counts, precision_matrix, 1
        except RRuntimeError as e:
            print(f"RRuntimeError: {e}")
            return selected_sub_idx, lambdax, np.zeros((p,p)), np.zeros((p,p)), 0


    def subsample_optimiser(self, b, Q, lambda_range):
        """
        Optimizes the objective function for all sub-samples and lambda values.
        Parameters
        ----------
        b : intpip 
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
        random.seed(42)
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

    assert Q > 0, "before call: Q must be greater than zero"
    assert J > 0, "before call: J must be greater than zero"

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
    psis = wp_tr_weights * Q # expansion to multipe prior sources: add a third dimension of length r, corresponding to the number of prior sources
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




def synthetic_run(p = 10, n = 500, b = 250, Q = 50, lambda_range = np.linspace(0.01, 0.05, 80)):
    # # Set random seed for reproducibility
    # np.random.seed(42)

    if p == 100:
        m = random.choice([1,2,2])
    elif p == 300:
        m = random.choice([3,5,6])
    elif p == 500:
        m = random.choice([5,8,10])
    elif p == 1000:
        m = random.choice([10,15,20])
    else:
        m = 5

    # TRUE NETWORK
    G = nx.barabasi_albert_graph(p, m, seed=42)
    adj_matrix = nx.to_numpy_array(G)
    
    # PRECISION MATRIX
    precision_matrix = -0.5 * adj_matrix

    # Add to the diagonal to ensure positive definiteness
    # Set each diagonal entry to be larger than the sum of the absolute values of the off-diagonal elements
    # in the corresponding row
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

    prior_matrix = np.zeros((p, p))

    # DATA MATRIX
    np.random.seed(42)
    data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)
    # print('data below \n')
    # print(data)

    # # MODEL RUN
    optimizer = SubsampleOptimizer(data, prior_matrix)
    edge_counts_all, success_counts, success_perc = optimizer.subsample_optimiser(b, Q, lambda_range)

    # lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)
    lambda_np = 0.078 
    p_k_matrix = np.zeros((p,p))

    opt_precision_mat = optimize_graph(data, prior_matrix, lambda_np)

    # # Estimate true p lambda_wp
    # lambda_wp, tau_tr, mus = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, 
    # prior_matrix)


    return opt_precision_mat, adj_matrix, edge_counts_all, success_counts, success_perc, lambda_np

def get_lambda_chunk(lambda_range, total_nodes, rank):
    """Return a chunk of lambda values based on the node index."""
    
    split_point = int(0.08 * len(lambda_range) / 0.4)  # find the index where 0.08 lies in lambda_range
    
    # Determine how many nodes will process the longer-running lambdas
    longer_lambda_nodes = 2 * total_nodes // 5
    if rank < longer_lambda_nodes:
        chunk_size = split_point // longer_lambda_nodes
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size
    else:
        chunk_size = (len(lambda_range) - split_point) // (total_nodes - longer_lambda_nodes)
        start_idx = split_point + (rank - longer_lambda_nodes) * chunk_size
        end_idx = start_idx + chunk_size

    return lambda_range[start_idx:end_idx]



def main(rank, size, use_full_lambda_range=False):
    #######################
    p = 100
    n = 500
    b = int(0.75 * n)
    Q = 100
    
    l_lo = 0 # 0.04050632911392405
    l_hi = 0.4 # 0.1569620253164557
    lambda_range = np.linspace(l_lo, l_hi, 40)
    #######################

    if not use_full_lambda_range:
        total_nodes = size
        # Get chunk of lambda values based on the node index
        lambda_range = get_lambda_chunk(lambda_range, total_nodes, rank)


    # run synthetic_run() for these values
    results = synthetic_run(p, n, b, Q, lambda_range)
    edge_counts_all = results[2]


    # When saving the results, handle the case where lambda_range might be empty
    try:
        lambda_min = lambda_range[0] if lambda_range.size > 0 else 'none'
        lambda_max = lambda_range[-1] if lambda_range.size > 0 else 'none'
        filename = f'net_results/results_{p,n,Q}_lamrange{lambda_min,lambda_max}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"Error saving results: {e}")

    return edge_counts_all,p,n,Q,lambda_range

if __name__ == "__main__":
    # Initialize MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Check if running in SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        edges,p,n,Q, lambda_range = main(rank=rank, size=size)

        try:
            all_edges = comm.gather(edges, root=0)
            if rank == 0:
                # Save combined results
                with open(f'net_results/combined_edge_counts_all.pkl', 'wb') as f:
                    pickle.dump(all_edges, f)

                # Transfer results to $HOME
                os.system(f"cp -r net_results/ $HOME/")
        except Exception as e:
            print(f"Error during MPI communication or file operations: {e}")

    else:
        # If no SLURM environment, run for entire lambda range
        edges, p, n, Q, lambda_range = main(rank=0, size=0,use_full_lambda_range=True)

        # save results to pickle file
        with open(f'net_results/edge_counts_all_{p,n,Q}_fullrange.pkl', 'wb') as f:
            pickle.dump(edges, f)
