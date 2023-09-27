import numpy as np
import math
from scipy.optimize import minimize
import scipy.stats as stats
from itertools import combinations
from sklearn.covariance import empirical_covariance

####### TODO #######
# Vectorize for loops
# Add unit tests
# Add Error Handling
# make separate l1_norms for edges with and without priors (They are different Thetas)
# implement "empirical null distribution similar to GSEA"
# check if the denominator of eq (12) is implemented correctly
# The function theta_k = post_distro.cdf(zk_vec[:,e_k] + epsilon) - post_distro.cdf(zk_vec[:,e_k] - epsilon), could index zk_vec incorrectly

def objective(theta, S, lambda_np, lambda_wp):
    # Reshape theta into a matrix
    p = int(np.sqrt(len(theta)))
    Theta = theta.reshape((p, p))
    Theta_np = Theta.copy()
    Theta_wp = Theta.copy()
    # Compute the objective function value
    log_det = np.log(np.linalg.det(Theta))
    trace_term = np.trace(np.dot(S, Theta))
    l1_norm_wp = np.sum(np.abs(Theta_wp))
    l1_norm_np = np.sum(np.abs(Theta_np))
    objective_value = -log_det + trace_term + lambda_np * l1_norm_np + lambda_wp * l1_norm_wp
    return objective_value

def subsampler(data, b, Q, lambda_range):
    n, p = data.shape
    # Generate all possible sub-samples without replacement
    all_subs = list(combinations(range(n), b))
    np.random.seed(42)
    selected_subs = np.random.choice(len(all_subs), min(Q, len(all_subs)), replace=False)
    
    edge_counts_all = np.zeros((p, p, len(lambda_range)))  # Initialize edge count matrix across lambdas and sub-samples

    # Loop for calculating graph structures across lambdas and sub-samples
    for l, lambdax in enumerate(lambda_range):
        edge_counts = np.zeros((p, p))  # Initialize edge count matrix for a given lambda
        
        for q in selected_subs:
            sub_sample = data[np.array(all_subs[q]), :]
            S = empirical_covariance(sub_sample)
            
            # Optimize the objective function with fixed lambda_wp (e.g., 0.1)
            initial_theta = np.eye(p).flatten()
            result = minimize(
                objective,
                initial_theta,
                args=(S, lambdax, lambdax),
                method='L-BFGS-B',
            )
            if result.success:
                Theta_opt = result.x.reshape((p, p))
                # Update edge count matrix
                edge_counts += (np.abs(Theta_opt) > 1e-5).astype(int)  # Assume edge exists if absolute value > 1e-5
            
        edge_counts_all[:,:,l] += edge_counts
    
    return edge_counts_all

def estimate_lambda_np(data, b, Q, lambda_range, edge_counts_all):
    n, p = data.shape
    results = []
    
    p_k_matrix = np.zeros((p, p))
    theta_matrix = np.zeros((p, p, len(lambda_range)))   # matrix of probabilities
    g_matrix = np.zeros((p, p, len(lambda_range)))       # Initialize instability matrix across lambdas
    zks = np.zeros((p, p, len(lambda_range)))           # Initialize edge count matrix across lambdas

    # Loop for calculating probabilities, instability, etc. across lambdas
    for l, lambda_np in enumerate(lambda_range):
        # Compute theta_k_lj for each edge
        theta_lj_matrix = np.zeros((p, p))               # matrix of probabilities
        g_l_matrix = np.zeros((p, p))                    # instability matrix for a given lambda
        for i in range(p):
            for j in range(p):
                z_k_lj = edge_counts_all[i, j, l]
                zks[i, j, l] = z_k_lj                   # store z_k_lj for later use
                N_k = np.sum(edge_counts_all[i, j, :])
                if l == 0:
                    p_k = N_k / (Q * len(lambda_range))  # Probability of edge presence
                    p_k_matrix[i, j] = p_k
                theta_lj_matrix[i, j] = math.comb(Q, z_k_lj) * (p_k ** z_k_lj) * ((1 - p_k) ** (Q - z_k_lj))
                f_k_lj = z_k_lj / Q
                g_l_matrix[i, j] = 4 * f_k_lj * (1 - f_k_lj)
        
        theta_matrix[:,:,l] += theta_lj_matrix
        g_matrix[:,:,l] += g_l_matrix                    # Update instability matrix across lambdas
    
    # Reshape the matrices for vectorized operations
    theta_matrix_reshaped = theta_matrix.reshape(len(lambda_range), -1)
    g_matrix_reshaped = g_matrix.reshape(len(lambda_range), -1)

    # Compute the score for each lambda_j using vectorized operations
    scores = np.sum(theta_matrix_reshaped * (1 - g_matrix_reshaped), axis=1)

    # Find the lambda_j that maximizes the score
    lambda_np = lambda_range[np.argmax(scores)]
    
    return lambda_np, p_k_matrix, zks



def estimate_lambda_wp(data, b, Q, p_k_matrix, zks, lambda_range, prior_matrix):
    n, p = data.shape
    results = []

    # reshape the prior matrix to only contain the edges in the lower triangle of the matrix
    wp_tr = [(i, j) for i, j in combinations(range(p), 2) if prior_matrix[i, j] != 0] # THIS SETS THE INDICES FOR ALL VECTORIZED OPERATIONS
    wp_tr_weights = [prior_matrix[comb[0], comb[1]] for comb in wp_tr]
    psis = wp_tr_weights * Q # expansion: add a third dimension of length r, corresponding to the number of prior sources

    p_k_vec = [p_k_matrix[comb[0], comb[1]] for comb in wp_tr]    
    z_mat = np.zeros((len(lambda_range), len(wp_tr))) # Stores zks for each edge across lambdas (shape: lambdas x edges)
    for l in range(len(lambda_range)):
        z_mat[l,:] =  [zks[comb[0], comb[1], l] for comb in wp_tr]

    # Alternative code
    # wp_tr_rows, wp_tr_cols = zip(*wp_tr)  # Unzip the wp_tr tuples into two separate lists
    # z_mat = zks[wp_tr_rows, wp_tr_cols, np.arange(len(lambda_range))[:, None]]


    # calculate mus, vars
    mus = [p_k * Q for p_k in p_k_vec]
    vars = [p_k * (1 - p_k) * Q for p_k in p_k_vec]

    tau_tr = np.sum(np.abs(mus - psis)) / len(wp_tr) # NOTE: alternatively, divide by np.sum(np.abs(wp_tr))


    ######## POSTERIOR DISTRIBUTION ########
    post_mus = np.zeros((len(wp_tr)))
    post_vars = np.zeros((len(wp_tr)))

    thetas = np.zeros((len(lambda_range), len(wp_tr))) # Initialize theta matrix across lambdas

    # Calculate a gaussian distribution with mean mu and variance var
    for e_k in range(len(wp_tr)):
        mu = mus[e_k]
        var = vars[e_k]
        # gaussian_d = norm(mu, var)

        psi = psis[e_k]
        # gaussian_p = norm(psi, tau_tr**2)

        post_mu = (mu * tau_tr**2 + psi * var) / (var + tau_tr**2)
        post_var = (var * tau_tr**2) / (var + tau_tr**2)
    
        # Create the normal distribution object
        post_distro = stats.norm(loc=post_mu, scale=np.sqrt(post_var))

        # Compute the CDF values at zk + epsilon and zk - epsilon
        epsilon = 1e-5
        thetas[:, e_k] = post_distro.cdf(z_mat[:,e_k] + epsilon) - post_distro.cdf(z_mat[:,e_k] - epsilon) # NOTE: WE MIGHT HAVE TO SEPARATELY CALCULATE FOR EACH LAMBDA

    # Frequency, instability, and score
    freq_mat = z_mat / Q                         # shape: lambdas x edges
    g_mat = 4 * freq_mat * (1 - freq_mat)

    # Scoring function
    scores = np.sum(thetas * (1 - g_mat), axis=1)

    # Find the lambda_j that maximizes the score
    lambda_wp = lambda_range[np.argmax(scores)]

    return lambda_wp, tau_tr, mus


def tau_permutations(data, tau_tr, prior_matrix, wp_tr, Q, mus):
    n, p = data.shape
    # Generate empirical null distribution of tau, similar to GSEA
    N_permutations = 10000
    tau_perm = np.zeros((N_permutations))

    num_edges = len(wp_tr)

    for _ in range(N_permutations):
        permuted_edges = [tuple(random.sample(range(n), 2)) for _ in range(num_edges)]
        permuted_weights = [prior_matrix[comb[0], comb[1]] for comb in permuted_edges]

        psi_perm = permuted_weights * Q

        tau_perm[_] = np.sum(np.abs(mus - psi_perm)) / num_edges
    
    tau_mean = np.mean(tau_perm)
    tau_perm_normalized = tau_perm / tau_mean
    tau_normalized = tau_tr / tau_mean

    # calculate percentage of taus that are greater than tau_tr, and subtract from 1 to get p-value
    p_value = 1 - (np.sum(tau_perm_normalized >= tau_normalized) / N_permutations)

    return p_value

