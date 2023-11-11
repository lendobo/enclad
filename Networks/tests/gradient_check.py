import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import empirical_covariance
import networkx as nx
from scipy.linalg import block_diag, eigh, inv
from numpy.random import multivariate_normal
from sklearn.covariance import empirical_covariance
from tqdm import tqdm
from scipy.optimize import fmin_l_bfgs_b



def objective(L_vector, S, lambdaboth, prior_matrix):
    # Cholesky: Reconstruct the lower triangular matrix L from the vector L_vector
    L = np.zeros((p, p))
    L[np.tril_indices(p)] = L_vector
    # Reconstruct the precision matrix P = LL^T
    precision_matrix = np.dot(L, L.T)
    det_value = np.linalg.det(precision_matrix)
    # if det_value <= 0 or np.isclose(det_value, 0):
    #     print("Warning: Non-invertible matrix")
    #     return np.inf  # return a high cost for non-invertible matrix
    
    log_det = np.log(det_value)
    trace_term = np.trace(np.dot(S, precision_matrix))
    base_objective = -log_det + trace_term

    # prior_entries = prior_matrix != 0
    # non_prior_entries = prior_matrix == 0
    # penalty_wp = lambda_wp * np.sum(np.abs(precision_matrix[prior_entries]))
    # penalty_np = lambda_np * np.sum(np.abs(precision_matrix[non_prior_entries]))

    # create a general pentalty term, which is the l1-norm of the precision matrix
    penalty = lambdaboth * np.sum(np.abs(precision_matrix))

    objective_value = base_objective + penalty # + penalty_np + penalty_wp
    return objective_value

def numerical_gradient(f, x, epsilon=1e-8):
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        unit_vector = np.zeros_like(x, dtype=float)
        unit_vector[i] = 1.0
        
        # Compute function value at x + epsilon * unit_vector
        f_plus = f(x + epsilon * unit_vector)
        
        # Compute function value at x - epsilon * unit_vector
        f_minus = f(x - epsilon * unit_vector)
        
        # Numerical gradient for dimension i
        grad[i] = (f_plus - f_minus) / (2 * epsilon)

    return grad

def callback_factory(S, lambdaboth, prior_matrix):
    def callback(precision_vector):
        numerical_grad = numerical_gradient(
            lambda precision_vector: objective(precision_vector, S, lambdaboth, prior_matrix),
            precision_vector
        )
        # print(f'Numerical Gradient at iteration: {numerical_grad}')
    return callback


def multiple_runs(num_runs=10, num_init_conds = 5, penalty=0.15, seed=42):
    all_results = []
    all_succ_ic = np.zeros(num_init_conds)
    for _ in range(num_runs):
        # Set a random seed for reproducibility
        np.random.seed(seed)

        # TRUE NETWORK
        G = nx.barabasi_albert_graph(p, 1, seed=seed)
        adj_matrix = nx.to_numpy_array(G)

        # PRECISION MATRIX
        precision_matrix = -0.5 * adj_matrix
        diagonal_values = 2 * np.abs(precision_matrix).sum(axis=1)
        np.fill_diagonal(precision_matrix, diagonal_values)
        eigenvalues = np.linalg.eigh(precision_matrix)[0]
        is_positive_definite = np.all(eigenvalues > 0)
        scaling_factors = np.sqrt(np.diag(precision_matrix))
        adjusted_precision = np.outer(1 / scaling_factors, 1 / scaling_factors) * precision_matrix
        covariance_mat = inv(adjusted_precision)

        # PRIOR MATRIX
        prior_matrix = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                if adj_matrix[i, j] != 0 and np.random.rand() < 0.95:
                    prior_matrix[i, j] = 0.9
                    prior_matrix[j, i] = 0.9
                elif adj_matrix[i, j] == 0 and np.random.rand() < 0.05:
                    prior_matrix[i, j] = 0.9
                    prior_matrix[j, i] = 0.9
        np.fill_diagonal(prior_matrix, 0)

        # DATA MATRIX
        data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)
        S = empirical_covariance(data)

        success_ic = np.zeros(num_init_conds)
        for ic in range(num_init_conds):

            # Initial guess
            Eye = np.zeros((p,p)) # 
            epsilon = 1e-1 + ic * 1e-1
            try:
                L_init = np.linalg.cholesky(inv(Eye + epsilon * np.eye(p)))
            except np.linalg.LinAlgError:
                print(f"Initial Guess {ic}: non-invertible matrix")
                continue

            # Convert L_init to a vector representing its unique elements
            initial_L_vector = L_init[np.tril_indices(p)]

            # Optimization using fmin_l_bfgs_b
            L_opt, f_min, info = fmin_l_bfgs_b(
                objective,
                initial_L_vector,
                args=(S, penalty, prior_matrix),
                approx_grad=True,    # Set to True to approximate the gradient
                epsilon=1e-8,        # Step size used to approximate the gradient
                callback = callback_factory(S, penalty, prior_matrix),
                pgtol=1e-5,          # You can adjust this for convergence criteria
                factr=1e7,           # Adjust for desired accuracy
                disp=0,              # Set to 1 for display
                maxiter=1000,        # Maximum iterations, adjust as needed
                bounds=[(None, None)] * len(initial_L_vector),  # No bounds on L_vector values
                maxls = 20
            )

            # Print diagnostic information
            print(f"-------------------\n {f_min}")
            print("Number of iterations:", info['nit'])
            print("Number of function calls:", info['funcalls'])
            # print("Gradient at solution:", info['grad'])
            print("Exit status:", info['task'])

            if info['warnflag'] == 0:
                print("Optimization converged successfully!")
                # print(f_min)
                # print(L_opt)

                L_tri_opt = np.zeros((p, p))
                L_tri_opt[np.tril_indices(p)] = L_opt
                # Compute the optimized precision matrix
                opt_precision_mat = np.dot(L_tri_opt, L_tri_opt.T)
                all_results.append(opt_precision_mat)
                success_ic[ic] = 1
            elif info['warnflag'] == 1:
                print("Warning: Too many function evaluations or iterations.")
                # print(f_min)
                # print(L_opt)
                success_ic[ic] = 1
            elif info['warnflag'] == 2:
                print("Warning: The optimization did not converge.")
                # print(f_min)
                # print(L_opt)



            # # Optimization WITH SCIPY.LINALG.MINIMIZE
            # result = minimize(
            #     objective,
            #     initial_L_vector,
            #     args=(S, penalty, prior_matrix),
            #     method='L-BFGS-B',
            #     options={'disp': False}  # Set to True to display convergence messages
            # )

            # # Store the result if the optimization was successful
            # if result.success:
            #     all_results.append(result.x)
        all_succ_ic += success_ic
    return all_results, adj_matrix, all_succ_ic

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
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Jaccard similarity
    jaccard_similarity = tp / (tp + fp + fn)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'jaccard_similarity': jaccard_similarity,
        'accuracy': accuracy
    }

    return metrics

# PARAMETERS
num_runs = 1
p = 10
n = 200
lambda_np, lambda_wp = [1.0, 1.0]

results, adj_matrix, all_succ_ic = multiple_runs(num_runs=num_runs, num_init_conds=20, penalty=0.3, seed=42)
print(all_succ_ic)
# print(results)
# print(len(results))

# # # Evaluate the results
# for result in results:
#     true_prec = -0.5 * adj_matrix
#     metrics = evaluate_reconstruction(adj_matrix, result)
#     print(metrics)

# for _ in tqdm(range(num_runs)):
#     # DATA MATRIX
#     data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)

#     S = empirical_covariance(data)

#     Eye = np.eye(p)

#     try:
#         epsilon = 1e-3
#         L_init = np.linalg.cholesky(inv(Eye + epsilon * np.eye(p)))
#     except np.linalg.LinAlgError:
#         print("Initial Guess: non-invertible matrix")

#     # Convert L_init to a vector representing its unique elements
#     initial_L_vector = L_init[np.tril_indices(p)]

#     # Compute the numerical gradient
#     numerical_grad = numerical_gradient(
#         lambda precision_vector: objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix),
#         initial_L_vector
#     )

#     ##  Print numerical gradient
#     # print(numerical_grad)

#     # Optimization with callback to check gradient at each iteration
#     result = minimize(
#         objective,
#         initial_L_vector,
#         args=(S, lambda_np, lambda_wp, prior_matrix),
#         method='L-BFGS-B',
#         callback=callback,
#         options={'disp': True}  # Set to True to display convergence messages
#     )

#     # Check for ABNORMAL_TERMINATION_IN_LNSRCH
#     if not result.success and 'ABNORMAL_TERMINATION_IN_LNSRCH' in result.message:
#         # Compute the gradient at the termination point
#         grad_at_failure = numerical_gradient(
#             lambda precision_vector: objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix),
#             result.x
#         )
        
#         # Store the gradient and the termination point
#         failed_gradients.append(grad_at_failure)
#         failed_points.append(result.x)
#         # print the function value F of result

# print(len(failed_gradients), len(failed_points))  # Number of failed runs

# # for fail in failed_gradients:
# #     print(fail)