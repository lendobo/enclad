import math
import networkx as nx

# Desired density
desired_density = 0.04

# An initial guess for the number of nodes
N_guess = 100

# Calculate the number of edges required for the guessed number of nodes to achieve the desired density
E_required = (desired_density * N_guess * (N_guess - 1)) / 2

N_guess, int(E_required)

# Initial number of nodes to start the network
m0 = 3

# Calculate m based on our guess
m = (E_required - (m0 * (m0 - 1) / 2)) / (N_guess - m0)
m = int(round(m))
print(f'density parameter for scale-free network: {m}')

# Generate the BA graph
G = nx.barabasi_albert_graph(25, 3)

# Check the actual number of nodes and edges in the generated graph
actual_N = G.number_of_nodes()
actual_E = G.number_of_edges()

actual_N, actual_E

# Calculate the actual density of the generated graph
actual_density = 2 * actual_E / (actual_N * (actual_N - 1))

print(actual_density)

# transalte it to an adjacency metraix
adj_matrix = nx.to_numpy_array(G)

# print(adj_matrix)
