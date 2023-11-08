import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from itertools import combinations
from scipy.special import comb, erf
from scipy.stats import norm

from old_piGGM import QJSweeper


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
    
    return lambda_np, theta_matrix



def estimate_lambda_wp(edge_counts_all, Q, lambda_range, prior_matrix):
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
    p, _, _ = edge_counts_all.shape
    J = len(lambda_range)

    N_k_matrix = np.sum(edge_counts_all, axis=2)
    p_k_matrix = N_k_matrix / (Q * J)

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
    post_mus = (mus * tau_tr**2 + psis * variances) / (variances + tau_tr**2)
    post_var = (variances * tau_tr**2) / (variances + tau_tr**2)

    # Since the normal distribution parameters are arrays...
    # Compute the CDF values directly using the formula for the normal distribution CDF
    epsilon = 1e-5
    z_scores_plus = (count_mat + epsilon - post_mus[None, :]) / np.sqrt(post_var)[None, :]
    z_scores_minus = (count_mat - epsilon - post_mus[None, :]) / np.sqrt(post_var)[None, :]
    
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

    # # print(scores)
    # print(tau_tr)
    # print(np.sum(p_k_vec) / len(p_k_vec))


    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]

    # print(lambda_wp)
    
    return lambda_wp, tau_tr, mus

# Load edge counts
filename_edges = 'net_results/combined_edge_counts_all_pnQ(50, 500, 300).pkl'
with open(filename_edges, 'rb') as f:
    edge_counts_all = pickle.load(f)

p = 50             # number of variables (nodes)
n = 500             # number of samples
b = int(0.75 * n)   # size of sub-samples
Q = 300             # number of sub-samples

lowerbound = 0.01
upperbound = 0.4
l_lo = 3     # Set it at knee-point index -1 
l_hi = 15    # set at knee-point index 


lambda_range = np.linspace(lowerbound, upperbound, 40)
select_lambda_range = lambda_range[l_lo:l_hi]

select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]

rank=1
size=1

# # Generate synthetic data and prior matrix
data, prior_matrix_t, adj_matrix = QJSweeper.generate_synth_data(p, n)

# make a random prior matrix with 0s and 1s
r_p_1 = np.random.randint(2, size=(p, p))
np.fill_diagonal(r_p_1, 0)

r_p_2 = np.random.randint(2, size=(p, p))
np.fill_diagonal(r_p_2, 0)


# check if r_p_1 and r_p_2 are the same
print(np.array_equal(r_p_1, r_p_2))

# # check if prior_matrix_r and prior_
# matrix_t are the same
# print(np.array_equal(prior_matrix_r, prior_matrix_t))

lambda_np, theta_mat = estimate_lambda_np(select_edge_counts_all, Q, select_lambda_range)

lambda_wp_t, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, prior_matrix_t)
lambda_wp_r, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, r_p_1)
lambda_wp_r2, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, r_p_2)

# print(lambda_np)
# print(lambda_wp_t)
# print(lambda_wp_r)