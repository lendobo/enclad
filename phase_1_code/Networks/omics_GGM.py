import numpy as np
import pandas as pd
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
from tqdm import tqdm
from concurrent.futures import as_completed
import sys
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
import time
import os

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
    model = GraphicalLasso(alpha=lambda_val, mode='cd', max_iter=100, tol=1e-3)
    try:
        model.fit(data)
        return model.precision_
    except Exception as e:
        print(f"Optimization did not succeed due to {str(e)}")
        return np.zeros((data.shape[1], data.shape[1]))


### Import data
selected_genes = 100
# cms4_data = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_midrange_.csv', index_col=0)
cms4_data = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_top{selected_genes}.csv', index_col=0)

cms2_data = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms2_top{selected_genes}.csv', index_col=0)

# downsizer = 100
# # subset the data at 10 random columns
# cms4_data = cms4_data.sample(n=downsizer, axis=1)
# cms2_data = cms2_data.sample(n=downsizer, axis=1)

cms2_array = cms2_data.values
# data_array = 2**data_array

network_size = cms2_array.shape[1]

# scale and center 
cms2_array = (cms2_array - cms2_array.mean(axis=0)) / cms2_array.std(axis=0)

prior_matrix = np.zeros((network_size, network_size))

# # # check QQ plot each gene (column) in a sqrt(n) x sqrt(n) grid
# network_size = data_array.shape[1]
# n_sqrt = int(np.sqrt(network_size))
# fig, axs = plt.subplots(n_sqrt, n_sqrt)
# for i in range(n_sqrt):
#     for j in range(n_sqrt):
#         axs[i, j].set_title(f"Gene {i*n_sqrt + j}")
#         stats.probplot(data_array[:, i*n_sqrt + j], plot=axs[i, j])
# plt.tight_layout()
# plt.show()




# Parameters
l_lo = 0 # 0.04050632911392405
l_hi = 0.5 # 0.1569620253164557
lambda_range = np.linspace(l_lo, l_hi, 20)

n = cms2_array.shape[0]
b = int(0.8 * n)   # [int(0.7 * n), int(0.75 * n), int(0.8 * n)]
Q = 300

print('network size: ', network_size)
print(f'number of samples: {n}, number of sub-samples: {Q}, sub-sample size: {b}')

################################# SYNTHETIC ######################################################

m = random.choice([1,2,2])
print(m)

# TRUE NETWORK
G = nx.barabasi_albert_graph(network_size, m, seed=42)  
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

# np.random.seed(42)
synth_data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)
print(f'shape of the synthetic data: {synth_data.shape}')

# # # check QQ plot each gene (column) in a sqrt(n) x sqrt(n) grid
# n = synth_data.shape[1]
# n_sqrt = int(np.sqrt(n))
# fig, axs = plt.subplots(n_sqrt, n_sqrt)
# for i in range(n_sqrt):
#     for j in range(n_sqrt):
#         axs[i, j].set_title(f"Gene {i*n_sqrt + j}")
#         stats.probplot(synth_data[:, i*n_sqrt + j], plot=axs[i, j])
# plt.tight_layout()
# plt.show()

################################### ########################################################

# # Network Inference, syntehtic data
# optimizer = SubsampleOptimizer(synth_data, prior_matrix)
# edge_counts_all, success_counts, success_perc = optimizer.subsample_optimiser(b, Q, lambda_range)
# lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)

# print('Synth Done')

# Network Inference, data
optimizer = SubsampleOptimizer(cms2_array, prior_matrix)
edge_counts_all, success_counts, success_perc = optimizer.subsample_optimiser(b, Q, lambda_range)
lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)

# save edge_counts_all to pkl file
with open(f'edge_counts_all_top{selected_genes}_lambda_{l_lo,l_hi}_Q{Q}_b{b}cms2.pkl', 'wb') as f:
    pickle.dump(edge_counts_all, f)