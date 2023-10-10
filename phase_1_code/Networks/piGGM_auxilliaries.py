####### TODO #######
# SWITCH TO PSEUDO LOG-LIKELIHOOD
# (mostly DONE) Vectorize the for loops 
# (DONE) implement parallelization for Q * J optimization runs 
# Once it runs, implement alternative and check if results are still the same
# check if the denominator of eq (12) is implemented correctly (Run both versions and compare)

###### Possible adjustments to experiment with #######
# Currently, we assume an edge is present if the optimized precision matrix has an absolute value > 1e-5.
# If the resulting network is too dense, we might have to change this threshold. However, the LASSO pushes many values to exactly 0, 
# so this might not be a problem.


def tau_permutations(data, tau_tr, prior_matrix, wp_tr, Q, mus, N_permutations=10000):
    """
    Calculates the p-value for the tau statistic.
    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    tau_tr : float
        The standard deviation of the prior distribution.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.
    wp_tr : array-like, shape (r, 2)
        The indices of the prior edges.
    Q : int
        The number of sub-samples.
    mus : array-like, shape (p, p)
        The mean of the prior distribution.
    N_permutations : int, optional
        The number of permutations to perform. The default is 10000.

    Returns
    -------
    p_value : float
        The p-value for the tau statistic.
    """
    n, p = data.shape
    # Generate empirical null distribution of tau, similar to GSEA
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


def test():
    """
    Tests the functions in this file.
    """
    data = np.random.rand(50, 10)
    b = 20
    Q = 3
    lambda_range = np.linspace(0.01, 1, 2)
    prior_matrix = np.random.rand(10, 10)
    
    # Test subsampler function
    edge_counts_all = subsample_optimiser(data, b, Q, lambda_range, prior_matrix)
    assert edge_counts_all.shape == (10, 10, 10), f"Expected (10, 10, 10), but got {edge_counts_all.shape}"
    
    # Test estimate_lambda_np function
    lambda_np, p_k_matrix, zks = estimate_lambda_np(data, b, Q, lambda_range, edge_counts_all)
    assert lambda_np in lambda_range, f"lambda_np not in lambda_range"
    assert p_k_matrix.shape == (10, 10), f"Expected (10, 10), but got {p_k_matrix.shape}"
    assert zks.shape == (10, 10, 10), f"Expected (10, 10, 10), but got {zks.shape}"
    
    # Test estimate_lambda_wp function
    lambda_wp, tau_tr, mus = estimate_lambda_wp(data, b, Q, p_k_matrix, zks, lambda_range, prior_matrix)
    assert lambda_wp in lambda_range, f"lambda_wp not in lambda_range"
    
    # Test tau_permutations function
    p_value = tau_permutations(data, tau_tr, prior_matrix, [(i, j) for i, j in combinations(range(10), 2)], Q, mus, N_permutations=100)
    assert 0 <= p_value <= 1, f"p_value out of range: {p_value}"
    
    print("All tests passed.")

# Demonstrating the prior penalty estimation
def generate_data_from_network(p=25):
    """
    Generate a data matrix from a synthetic network.
    Returns the adjacency matrix (representing the true network) and the data matrix.
    """
    # Generate a Barabasi-Albert graph
    G = nx.barabasi_albert_graph(p, 3)
    adj_matrix = nx.to_numpy_array(G)

    # Ensure the matrix is positive definite
    min_eig = np.min(np.real(eigh(adj_matrix)[0]))
    if min_eig < 0:
        adj_matrix -= 10 * min_eig * np.eye(adj_matrix.shape[0])

    # Derive the covariance matrix from the adjacency matrix and generate data
    covariance_matrix = inv(adj_matrix)
    data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_matrix, size=50)

    return adj_matrix, data

def modify_prior_matrix(true_prior, random_prior, perc_replacements):
    """
    Modify the true prior matrix by replacing specified number of fields with the random prior matrix.
    """
    p = true_prior.shape[0]
    modified_prior = true_prior.copy()

    num_replacements = int(perc_replacements * p * (p - 1) / 2)

    # Get the indices of the fields to be replaced
    indices = np.array([(i, j) for i in range(p) for j in range(p)])
    selected_indices = indices[np.random.choice(indices.shape[0], num_replacements, replace=False)]

    # Replace the fields in the true prior matrix with the fields from the random prior matrix
    for idx in selected_indices:
        modified_prior[idx[0], idx[1]] = random_prior[idx[0], idx[1]]
        modified_prior[idx[1], idx[0]] = random_prior[idx[1], idx[0]]  # Keep the matrix symmetric

    return modified_prior


# PARAMS
p = 5
n = 50
b = 25
Q = 10
lambda_range = np.linspace(0.01, 0.2, 5)

# Generate a single data matrix from a synthetic network
adj_matrix, data = generate_data_from_network(p)

# Generate the true prior matrix and a random matrix
true_prior = adj_matrix.copy()
np.fill_diagonal(true_prior, 0)
random_prior = np.random.randint(2, size=(25, 25))
np.fill_diagonal(random_prior, 0)

# Generate a sequence of modified prior matrices with increasing number of replacements
replacements = [.9, .5, .1, .05, .01]
modified_priors = [modify_prior_matrix(true_prior, random_prior, r) for r in replacements]

modified_priors.append(true_prior)  # Display the first modified prior matrix for reference

print(adj_matrix)
for prior_matrix in modified_priors:
    print(prior_matrix)

# # Estimate lambda_wp for each prior matrix
# lambda_wp_values = []
# for prior_matrix in modified_priors:
#     # Run optimization
#     optimizer = SubsampleOptimizer(data, prior_matrix)
#     edge_counts_all = optimizer.subsample_optimiser(b, Q, lambda_range)
#     lambda_np, p_k_matrix, theta_matrix = estimate_lambda_np(edge_counts_all, Q, lambda_range)
#     lambda_wp, _, _ = estimate_lambda_wp(data, Q, p_k_matrix, edge_counts_all, lambda_range, prior_matrix)
#     lambda_wp_values.append(lambda_wp)

# print(lambda_wp_values)