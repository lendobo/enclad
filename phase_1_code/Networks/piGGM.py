import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from sklearn.covariance import empirical_covariance

def objective(theta, S, lambda_np, lambda_wp):
    # Reshape theta into a matrix
    p = int(np.sqrt(len(theta)))
    Theta = theta.reshape((p, p))
    # Compute the objective function value
    log_det = np.log(np.linalg.det(Theta))
    trace_term = np.trace(np.dot(S, Theta))
    l1_norm = np.sum(np.abs(Theta))
    objective_value = -log_det + trace_term + lambda_np * l1_norm + lambda_wp * l1_norm
    return objective_value

def run_ggm(data, b, Q, lambda_range):
    n, p = data.shape
    results = []

    # Generate all possible sub-samples without replacement
    all_subs = list(combinations(range(n), b))
    selected_subs = np.random.choice(len(all_subs), min(Q, len(all_subs)), replace=False)
    
    edge_counts_all = np.zeros((p, p, len(lambda_range)))  # Initialize edge count matrix across lambdas and sub-samples

    # Loop for calculating graph structures across lambdas and sub-samples
    for l, lambda_np in enumerate(lambda_range):
        edge_counts = np.zeros((p, p))  # Initialize edge count matrix for a given lambda
        
        for q in selected_subs:
            sub_sample = data[np.array(all_subs[q]), :]
            S = empirical_covariance(sub_sample)
            
            # Optimize the objective function with fixed lambda_wp (e.g., 0.1)
            initial_theta = np.eye(p).flatten()
            result = minimize(
                objective,
                initial_theta,
                args=(S, lambda_np, 0.1),
                method='L-BFGS-B',
            )
            if result.success:
                Theta_opt = result.x.reshape((p, p))
                # Update edge count matrix
                edge_counts += (np.abs(Theta_opt) > 1e-5).astype(int)  # Assume edge exists if absolute value > 1e-5
            
        edge_counts_all[:,:,l] += edge_counts
    
    theta_matrix = np.zeros((p, p, len(lambda_range))) # matrix of probabilities
    g_matrix = np.zeros((p, p, len(lambda_range)))  # Initialize instability matrix across lambdas

    # Loop for calculating probabilities, instability, etc. across lambdas
    for l, lambda_np in enumerate(lambda_range):
        # Compute theta_k_lj for each edge
        theta_lj_matrix = np.zeros((p, p)) # matrix of probabilities
        g_lj_matrix = np.zeros((p, p)) # instability matrix
        for i in range(p):
            for j in range(p):
                z_k_lj = edge_counts_all[i, j, l]
                N_k = np.sum(edge_counts_all[i, j, :])
                p_k = N_k / (Q * len(lambda_range))  # Probability of edge presence
                theta_lj_matrix[i, j] = comb(Q, z_k_lj) * (p_k ** z_k_lj) * ((1 - p_k) ** (Q - z_k_lj))
                f_k_lj = z_k_lj / Q
                g_lj_matrix[i, j] = 4 * f_k_lj * (1 - f_k_lj)
        
        theta_matrix[:,:,l] += theta_lj_matrix
        g_matrix[:,:,l] += g_k_lj_matrix
    
    # Reshape the matrices for vectorized operations
    theta_matrix_reshaped = theta_matrix.reshape(len(lambda_range), -1)
    g_matrix_reshaped = g_matrix.reshape(len(lambda_range), -1)

    # Compute the score for each lambda_j using vectorized operations
    score_lj = np.sum(theta_matrix_reshaped * (1 - g_matrix_reshaped), axis=1)

    # Find the lambda_j that maximizes the score
    lambda_np = lambda_range[np.argmax(score_lj)]

    
    # Store theta_k_lj_matrix or other results if necessary
    results.append({
        'lambda_np': lambda_np,
        'lambda_wp': 1,
        # ...
    })
    
    return results

# Example usage:
# lambda_range = np.linspace(0.01, 1, 10)
# results = run_ggm(data, b=50, Q=100, lambda_range=lambda_range)

