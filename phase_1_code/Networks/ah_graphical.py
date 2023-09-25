# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
from numpy.random import multivariate_normal


# %% Synthetic network generation
# Parameters
nodes = 300

# Generate a scale-free network using the Barabási-Albert model
# The number of edges to attach from a new node to existing nodes is set to 2 to control the network density
BA_graph = nx.barabasi_albert_graph(nodes, 4)

# # Display the network
# plt.figure(figsize=(10, 10))
# nx.draw(BA_graph, node_size=50, node_color='blue', with_labels=False)
# plt.title('Scale-Free Network (Barabási-Albert model)')
# plt.show()

# Calculate network density
edges = BA_graph.number_of_edges()
max_possible_edges = nodes * (nodes - 1) / 2  # max edges for an undirected graph with n nodes
network_density = (edges / max_possible_edges) * 100

print(edges, network_density)

# %% PRECISION MATRIX

# Get the adjacency matrix (will be used as the precision matrix)
precision_matrix = nx.adjacency_matrix(BA_graph).todense().astype(float)

# Add identity to the precision matrix to ensure it's invertible
# This corresponds to adding self-loops in the graph, which doesn't change the graph structure significantly
precision_matrix += np.eye(nodes)

# Invert the precision matrix to get the covariance matrix
covariance_matrix = inv(precision_matrix)

# Generate synthetic data (e.g., 1000 samples)
n_samples = 1000
synthetic_data = multivariate_normal(mean=np.zeros(nodes), cov=covariance_matrix, size=n_samples)

# The true precision matrix is the matrix we used plus the identity matrix
true_precision_matrix = precision_matrix

# Show the first 5 rows of synthetic data
synthetic_data[:5]


# %% PARTIAL CORRELATION

def calculate_partial_correlation(precision_matrix):
    """
    Calculate the partial correlation matrix from the precision matrix.
    """
    # Diagonal elements of the precision matrix
    diag_elements = np.sqrt(np.diag(precision_matrix))
    
    # Outer product of the diagonal elements
    diag_outer_product = np.outer(diag_elements, diag_elements)
    
    # Calculate partial correlation matrix
    partial_correlation_matrix = -precision_matrix / diag_outer_product
    
    # Set the diagonal elements to 0 as partial correlation of a variable with itself is not defined
    np.fill_diagonal(partial_correlation_matrix, 0)
    
    return partial_correlation_matrix

# Calculate partial correlation matrix
partial_correlation_matrix = calculate_partial_correlation(true_precision_matrix)

# Calculate absolute value of the partial correlation matrix for the PPI prior matrix
PPI_prior_matrix = np.abs(partial_correlation_matrix)

# Show a portion of the PPI prior matrix (first 5x5 block)
PPI_prior_matrix[:5, :5]

# Count the number of edges in the PPI prior matrix
PPI_edges = np.count_nonzero(PPI_prior_matrix) / 2

# Calculate the network density of the PPI prior matrix
PPI_network_density = (PPI_edges / max_possible_edges) * 100
print(PPI_edges, PPI_network_density)

# %%
