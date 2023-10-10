import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import inv, eigh
from numpy.random import multivariate_normal
from itertools import combinations

p = 10
n = 500

# TRUE NETWORK
G = nx.barabasi_albert_graph(p, 1, seed=1)
adj_matrix = nx.to_numpy_array(G)
print(adj_matrix)

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