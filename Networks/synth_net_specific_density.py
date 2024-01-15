import math
import networkx as nx
import numpy as np

# Desired density
desired_densities = [0.02, 0.03, 0.04]
desired_densities = [0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]


# Desired node sizes
node_sizes = [100, 150, 300, 500, 750, 1000]

results = np.zeros((len(node_sizes), len(desired_densities)))

m0 = 3

for desired_density in desired_densities:
    for N_guess in node_sizes:
        i = desired_densities.index(desired_density)
        j = node_sizes.index(N_guess) 
        # Calculate the number of edges required for the guessed number of nodes to achieve the desired density
        E_required = (desired_density * N_guess * (N_guess - 1)) / 2

        # Calculate m based on our guess
        m = (E_required - (m0 * (m0 - 1) / 2)) / (N_guess - m0)
        m = int(round(m))
        # print(f'density parameter for scale-free network: {m}')

        # Generate the BA graph
        G = nx.barabasi_albert_graph(N_guess, m)

        # Check the actual number of nodes and edges in the generated graph
        actual_N = G.number_of_nodes()
        actual_E = G.number_of_edges()

        # print(f'nodes and edges: {actual_N, actual_E}')

        # Calculate the actual density of the generated graph
        actual_density = 2 * actual_E / (actual_N * (actual_N - 1))

        # print(actual_density)

        # results.append((actual_N, actual_E, actual_density, m))
        results[j][i] = m

# # print the output for each node size tested
# for i in range(len(node_sizes)):
#     print(f'nodes: {results[i][0]}, edges: {results[i][1]}, density: {results[i][2]}, density parameter: {results[i][3]}')

# print the density parameter for each desired_density and node size
for i in range(len(node_sizes)):
    print(f'node size: {node_sizes[i]}')
    for j in range(len(desired_densities)):
        print(f'desired density: {desired_densities[j]}, density parameter: {results[i][j]}')
    print('\n')
