import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from itertools import combinations
from scipy.special import comb, erf
from scipy.stats import norm
from scipy.optimize import curve_fit
import warnings
from tqdm import tqdm

from piglasso import QJSweeper



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
    for i, p_k in enumerate(p_k_vec):
        if p_k < 1e-5:
            p_k_vec[i] = 1e-5

    count_mat = np.zeros((J, len(wp_tr_idx))) # Stores counts for each edge across lambdas (shape: lambdas x edges)
    for l in range(J):
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
    # print(scores)

    # # print(scores)
    # print(tau_tr)
    # print(np.sum(p_k_vec) / len(p_k_vec))


    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]

    # print(lambda_wp)
    
    return lambda_wp, tau_tr, mus


# Define a linear function for curve fitting
def linear_func(x, a, b):
    return a * x + b

def fit_lines_and_get_error(index, lambdas, edge_counts, left_bound, right_bound):
    # Only consider data points within the specified bounds
    left_data = lambdas[left_bound:index+1]
    right_data = lambdas[index:right_bound]

    if len(left_data) < 3 or len(right_data) < 3:
        return np.inf

    # Fit lines to the left and right of current index within bounds
    # print(index)
    params_left, _ = curve_fit(linear_func, left_data, edge_counts[left_bound:index+1])
    # print(index)
    params_right, _ = curve_fit(linear_func, right_data, edge_counts[index:right_bound])
    
    # Calculate fit errors within bounds
    error_left = np.sum((linear_func(left_data, *params_left) - edge_counts[left_bound:index+1]) ** 2)
    error_right = np.sum((linear_func(right_data, *params_right) - edge_counts[index:right_bound]) ** 2)
    
    return error_left + error_right

def find_knee_point(lambda_range, edge_counts_all, left_bound, right_bound):
    errors = [fit_lines_and_get_error(i, lambda_range, edge_counts_all, left_bound, right_bound) 
              for i in range(left_bound, right_bound)]
    knee_point_index = np.argmin(errors) + left_bound
    return knee_point_index

def find_all_knee_points(lambda_range, edge_counts_all):
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    # Find the main knee point across the full range
    main_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, len(lambda_range))
    main_knee_point = lambda_range[main_knee_point_index]
    
    # For the left knee point, consider points to the left of the main knee point
    left_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, main_knee_point_index)
    left_knee_point = lambda_range[left_knee_point_index]
    
    # For the right knee point, consider points to the right of the main knee point
    # Update the bounds to ensure the fit_lines_and_get_error function considers only the right subset
    right_knee_point_index = find_knee_point(lambda_range, edge_counts_all, main_knee_point_index, len(lambda_range))
    right_knee_point = lambda_range[right_knee_point_index]
    
    return left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, main_knee_point_index, right_knee_point_index

# Main code
if __name__ == "__main__":
    #### Main code ####
    p = -1             # number of variables (nodes)
    n = 500             # number of samples
    b = int(0.75 * n)   # size of sub-samples
    Q = 300             # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.4
    granularity = 40
    lambda_range = np.linspace(lowerbound, upperbound, 40)

    filename_edges = f'Networks/net_results/local_omics_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_{granularity}.pkl'
    with open(filename_edges, 'rb') as f:
        edge_counts_all = pickle.load(f)

    # divide each value in edge_counts_all by 2*Q
    edge_counts_all = edge_counts_all / (2 * Q)


    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)
    print("Left Knee Point at lambda =", left_knee_point)
    print("Main Knee Point at lambda =", main_knee_point)
    print("Right Knee Point at lambda =", right_knee_point)

    # We will now plot the additional lines: the right red line and the left magenta line
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    plt.figure(figsize=(14, 7))
    plt.plot(lambda_range, edge_counts_all, 'bo', label='Edge Counts', alpha = 0.5)

    # Fit and plot the lines for the left knee point
    left_data = lambda_range[:left_knee_point_index+1]
    left_fit_params, _ = curve_fit(linear_func, left_data, edge_counts_all[:left_knee_point_index+1])
    plt.plot(left_data, linear_func(left_data, *left_fit_params), 'r-', label='Left Fit')

    # Fit and plot the line between the left knee point and the main knee point (right red line)
    left_knee_to_main_data = lambda_range[left_knee_point_index:knee_point_index+1]
    left_knee_to_main_fit_params, _ = curve_fit(linear_func, left_knee_to_main_data, edge_counts_all[left_knee_point_index:knee_point_index+1])
    plt.plot(left_knee_to_main_data, linear_func(left_knee_to_main_data, *left_knee_to_main_fit_params), 'r--', label='Right of Left Knee Fit')

    # Fit and plot the lines for the main knee point
    main_left_data = lambda_range[:knee_point_index]
    main_right_data = lambda_range[knee_point_index:]
    main_left_fit_params, _ = curve_fit(linear_func, main_left_data, edge_counts_all[:knee_point_index])
    main_right_fit_params, _ = curve_fit(linear_func, main_right_data, edge_counts_all[knee_point_index:])
    plt.plot(main_left_data, linear_func(main_left_data, *main_left_fit_params), 'g-', label='Main Left Fit')
    plt.plot(main_right_data, linear_func(main_right_data, *main_right_fit_params), 'g-', label='Main Right Fit')

    # Fit and plot the line between the main knee point and the right knee point (left magenta line)
    main_to_right_knee_data = lambda_range[knee_point_index:right_knee_point_index+1]
    main_to_right_knee_fit_params, _ = curve_fit(linear_func, main_to_right_knee_data, edge_counts_all[knee_point_index:right_knee_point_index+1])
    plt.plot(main_to_right_knee_data, linear_func(main_to_right_knee_data, *main_to_right_knee_fit_params), 'm--', label='Left of Right Knee Fit')

    # Fit and plot the lines for the right knee point
    right_data = lambda_range[right_knee_point_index:]
    right_fit_params, _ = curve_fit(linear_func, right_data, edge_counts_all[right_knee_point_index:])
    plt.plot(right_data, linear_func(right_data, *right_fit_params), 'm-', label='Right Fit')

    # Mark the knee points on the plot
    plt.axvline(x=left_knee_point, color='r', linestyle='--', label='Left Knee Point')
    plt.axvline(x=main_knee_point, color='g', linestyle='--', label='Main Knee Point')
    plt.axvline(x=right_knee_point, color='m', linestyle='--', label='Right Knee Point')

    plt.xlabel('Lambda')
    plt.ylabel('Edge Counts')
    plt.title('Knee Points and Fitted Lines')
    plt.legend()
    plt.grid()
    plt.show()



