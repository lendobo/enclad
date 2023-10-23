import math
import networkx as nx

# Desired density
desired_density = 0.04

# An initial guess for the number of nodes
N_guess = 300

# Desired node sizes
node_sizes = [50, 100, 300, 500, 1000]

results = []

m0 = 3

for N_guess in node_sizes:
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

    results.append((actual_N, actual_E, actual_density, m))

# print the output for each node size tested
for i in range(len(node_sizes)):
    print(f'nodes: {results[i][0]}, edges: {results[i][1]}, density: {results[i][2]}, density parameter: {results[i][3]}')
