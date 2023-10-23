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
            print('success')
            # Convert result.x back to a lower triangular matrix
            L_opt = np.zeros((p, p))
            L_opt[np.tril_indices(p)] = result.x
            # Compute the optimized precision matrix
            opt_precision_mat = np.dot(L_opt, L_opt.T)
            edge_counts = (np.abs(opt_precision_mat) > 1e-5).astype(int)
            return selected_sub_idx, lambdax, edge_counts
        else:
            print('failure')
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

    # # Writing to a file for diagnosis
    # original_stdout = sys.stdout
    # with open(f'out/eval_diagnosis.txt', 'w') as f:
    #     sys.stdout = f
    #     for e in range(len(wp_tr_idx)):
    #         print('\n\n\n')
    #         for j in range(len(lambda_range)):
    #             print('\n')
    #             print(f'lambda: {lambda_range[j]}')
    #             print(f'z_k_{str(e)}: {count_mat[j, e]}')
    #             print(f'prior_mu: {psis[e]}')
    #             print(f'data_mu: {mus[e]}')
    #             print(f'posterior_mu: {post_mu[e]}')
    #             print('posterior var: ', post_var[e])
    #             print(f'thetas: {thetas[j, e]}')
    #             print(f'g_mat: {g_mat[j, e]}')
    #             print(f'scores: {scores[j]}')
    # sys.stdout = original_stdout

    return lambda_wp, tau_tr, mus











from sklearn.covariance import GraphicalLasso, empirical_covariance

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




def synthetic_run(p = 50, n = 500, b = 250, Q = 50, lambda_range = np.linspace(0.01, 0.05, 20)):
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
            if adj_matrix[i, j] != 0 and np.random.rand() < 0.9 :
                prior_matrix[i, j] = 1
                prior_matrix[j, i] = 1
            elif adj_matrix[i, j] == 0 and np.random.rand() < 0.1:
                prior_matrix[i, j] = 1
                prior_matrix[j, i] = 1
    np.fill_diagonal(prior_matrix, 0)

    # DATA MATRIX
    data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)

    # MODEL RUN
    optimizer = SubsampleOptimizer(data, prior_matrix)
    edge_counts_all = optimizer.subsample_optimiser(b, Q, lambda_range)
    print(f'check0: edge_counts_all: {edge_counts_all[0, 0, 0]}')
    lambda_np, p_k_matrix, _ = estimate_lambda_np(edge_counts_all, Q, lambda_range)
    print(f'check1: lambda_np: {lambda_np}')
    # Estimate true p lambda_wp
    lambda_wp, _, _ = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix)
    print(f'check2: lambda_wp: {lambda_wp}')

    opt_precision_mat = optimize_graph(data, prior_matrix, lambda_np, lambda_wp)
    print(f'check3: opt_precision_mat: {opt_precision_mat}')

    return lambda_np, lambda_wp, opt_precision_mat, adj_matrix


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


# Example usage
num_runs = 100
p = 10
n = 500
b = int(n / 2)
Q = 50
lambda_granularity = 15
lambda_range = np.linspace(0.03, 0.06, lambda_granularity)

# sample run
lambda_np, lambda_wp, opt_precision_mat, adj_matrix = synthetic_run(p=p, n=n, b=b, Q=Q, lambda_range=lambda_range)
print("lambda_np:", lambda_np)
print("lambda_wp:", lambda_wp)

# sample evaluation
metrics = evaluate_reconstruction(adj_matrix, opt_precision_mat)
print("metrics:", metrics)





# lambda_np_values = []
# lambda_wp_values = []
# random_lambda_wp_values = []
# edge_counts_all_runs = []

# # Set the filenames based on initial parameters
# lambda_wp_filename = f'out/lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'
# random_lambda_wp_filename = f'out/random_lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'

# for _ in tqdm(range(num_runs)):
#     lambda_np, lambda_wp, random_lambda_wp, edge_counts_all = synthetic_run(p=p, n=n, b=b, Q=Q, lambda_range=lambda_range)
    

#     lambda_np_values.append(lambda_np)
#     lambda_wp_values.append(lambda_wp)
#     random_lambda_wp_values.append(random_lambda_wp)
#     edge_counts_all_runs.append(edge_counts_all)

#     # save lambda_wp values and random_lambda_wp values to csv, every 10 runs
#     if _ % 10 == 0:
#         with open(lambda_wp_filename, 'a') as f:
#             np.savetxt(f, lambda_wp_values, delimiter=',')
        
#         with open(random_lambda_wp_filename, 'a') as f:
#             np.savetxt(f, random_lambda_wp_values, delimiter=',')

# # RESULTS
# mean_lambda_np = np.mean(lambda_np_values)
# mean_lambda_wp = np.mean(lambda_wp_values)
# mean_random_lambda_wp = np.mean(random_lambda_wp_values)

# print("Mean lambda_np:", mean_lambda_np)
# print("Mean lambda_wp:", mean_lambda_wp)
# print("Mean random lambda_wp:", mean_random_lambda_wp)

# # Save the results to csv
# mean_lambda_wp_filename = f'out/mean_lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'
# mean_random_lambda_wp_filename = f'out/mean_random_lambda_wp_values_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'
# edge_counts_all_runs_filename = f'out/edge_counts_all_runs_{str(lambda_range[0])} - {str(lambda_range[1]), str(n)}_gran{lambda_granularity}.csv'

# with open(mean_lambda_wp_filename, 'a') as f:
#     np.savetxt(f, [mean_lambda_wp], delimiter=',')
# with open(mean_random_lambda_wp_filename, 'a') as f:
#     np.savetxt(f, [mean_random_lambda_wp], delimiter=',')
# # Flatten and stack edge_counts_all_runs
# stacked_edge_counts_all_runs = np.vstack([arr.flatten() for arr in edge_counts_all_runs])
# with open(edge_counts_all_runs_filename, 'a') as f:
#     np.savetxt(f, stacked_edge_counts_all_runs, delimiter=',')
# # To open and reshape back into original shape:
# # loaded_data = np.loadtxt(filename, delimiter=',')
# # list_of_reshaped_arrays = [row.reshape((p, p, J)) for row in loaded_data]
