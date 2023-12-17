# %%
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import math
import random
import copy
import time
from mpi4py import MPI
import pickle as pkl
import os
# import json
# import h5py
from tqdm import tqdm
import argparse
import sys

# %%
# Check if the script is running in an environment with predefined sys.argv (like Jupyter or certain HPC environments)
if 'ipykernel_launcher.py' in sys.argv[0] or 'mpirun' in sys.argv[0]:
    # Create a list to hold the arguments you want to parse
    args_to_parse = []

    # Iterate through the system arguments
    for arg in sys.argv:
        # Add only your specified arguments to args_to_parse
        if '--koh' in arg or '--kob' in arg or '--cms' in arg:
            args_to_parse.extend(arg.split('='))
else:
    args_to_parse = sys.argv[1:]  # Exclude the script name

# Command Line Arguments
parser = argparse.ArgumentParser(description='Run QJ Sweeper with command-line arguments.')
parser.add_argument('--koh', type=int, default=5, help='Number of hub nodes to knock out')
parser.add_argument('--kob', type=int, default=5, help='Number of bottom nodes to knock out')
parser.add_argument('--red_range', type=str, default='0.05,0.05,1', help='Range of reduction factors to investigate')
parser.add_argument('--cms', type=str, default='cmsALL', choices=['cmsALL', 'cms123'], help='CMS to use')
parser.add_argument('--mode', type=str, default='disruption', choices=['disruption', 'transition'], help='Type of knockout analysis')

args = parser.parse_args(args_to_parse)


# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if "SLURM_JOB_ID" not in os.environ:
    rank = 0
    size = 1

# %%
# Adjust Laplacian Matrix Calculation for Weighted Graph
def weighted_laplacian_matrix(G):
    """
    Calculate the Laplacian matrix for a weighted graph.
    """
    # Weighted adjacency matrix
    W = nx.to_numpy_array(G, weight='weight')
    # Diagonal matrix of vertex strengths
    D = np.diag(W.sum(axis=1))
    # Weighted Laplacian matrix
    L = D - W
    return L



def laplacian_exponential_kernel_eigendecomp(L, t):
    """
    Function to compute the Laplacian exponential diffusion kernel using eigen-decomposition
    The function takes the Laplacian matrix L and a time parameter t as inputs.
    """
    # Calculate the eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Compute the matrix exponential using eigenvalues
    exp_eigenvalues = np.exp(-t * eigenvalues)
    # Reconstruct the matrix using the eigenvectors and the exponentiated eigenvalues
    kernel = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T

    return kernel

def laplacian_exponential_diffusion_kernel(L, t):
    """
    compute the Laplacian exponential kernel for a given t value"""
    return scipy.linalg.expm(-t * L)

def knockout_node(G, node_to_isolate):
    """
    Isolates a node in the graph by removing all edges connected to it.
    
    :param G: NetworkX graph
    :param node_to_isolate: Node to isolate
    :return: None
    """
    modified_graph = G.copy()
    # Remove all edges to and from this node
    edges_to_remove = list(G.edges(node_to_isolate))
    modified_graph.remove_edges_from(edges_to_remove)
    new_laplacian = weighted_laplacian_matrix(modified_graph)

    return modified_graph, new_laplacian

def knockdown_node_both_layers(G, node_to_isolate_base, reduction_factor=0.3):
    """
    Reduces the weights of all edges connected to a node in both layers of the graph.

    :param G: NetworkX graph
    :param node_to_isolate_base: Base node name whose edges will be reduced in both layers
    :param node_to_isolate_base: Base node name whose edges will be reduced in both layers
    :param reduction_factor: Factor to reduce edge weights by, defaults to 0.5
    :return: Tuple containing the modified graph and its weighted Laplacian matrix
    """

    modified_graph = G.copy()
    
    # Add layer suffixes to the base node name
    node_to_isolate_proteomics = f"{node_to_isolate_base}.p"
    node_to_isolate_transcriptomics = f"{node_to_isolate_base}.t"
    
    # Reduce the weight of all edges to and from this node in both layers
    for node_to_isolate in [node_to_isolate_proteomics, node_to_isolate_transcriptomics]:
        for neighbor in G[node_to_isolate]:
            modified_graph[node_to_isolate][neighbor]['weight'] = reduction_factor
            modified_graph[neighbor][node_to_isolate]['weight'] = reduction_factor
    
    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = weighted_laplacian_matrix(modified_graph)
    return modified_graph, new_laplacian


# %%                  ############################################# double5 DEMO NET#########################

def double5_demonet(inter_layer_weight=1.0):
    # Create a new graph
    G = nx.Graph()
    # Define nodes
    nodes_A = [f"{i}.p" for i in range(1, 6)]
    nodes_B = [f"{i}.t" for i in range(1, 6)]
    # Add nodes
    G.add_nodes_from(nodes_A)
    G.add_nodes_from(nodes_B)
    # Add edges to form a complete circle in each layer
    for i in range(5):
        G.add_edge(nodes_A[i], nodes_A[(i-1)%5])  # A layer
        G.add_edge(nodes_B[i], nodes_B[(i-1)%5])  # B layer
    # Add edges between corresponding nodes of different layers
    for i in range(5):
        G.add_edge(nodes_A[i], nodes_B[i], layer='interlayer')
    # Add manual edges
    G.add_edge("1.t", "3.t")
    G.add_edge("1.p", "4.p")

    weighted_G = G.copy()
    for u, v, data in weighted_G.edges(data=True):
        if data.get('layer') == 'interlayer':
            weighted_G[u][v]['weight'] = inter_layer_weight
        else:
            weighted_G[u][v]['weight'] = 1.0


    # get pandas adjacency
    adj = nx.to_pandas_adjacency(weighted_G)

    lap = weighted_laplacian_matrix(weighted_G)
    # roudn values of lap to 3 decimal places
    lap = np.round(lap, 1)

    return weighted_G, adj, lap

# double5_net, double5_adj, double5_lap = double5_demonet()
# # get degrees
# degrees = [val for (node, val) in double5_net.degree()]
# knockdown_double5_g, knockdown_double5_lap = knockdown_node_both_layers(double5_net, '1', reduction_factor=0.5)
# double5_lap


def create_multiplex_graph(num_nodes, inter_layer_weight=1.0):
    """
    Creates a multiplex graph with two layers, each having the specified number of nodes.
    """
    G = nx.Graph()

    # Define nodes for each layer
    nodes_layer_p = [f"{i}.p" for i in range(num_nodes)]
    nodes_layer_t = [f"{i}.t" for i in range(num_nodes)]

    # Add nodes
    G.add_nodes_from(nodes_layer_p)
    G.add_nodes_from(nodes_layer_t)

    # Add random edges within each layer
    for _ in range(num_nodes * 2):  # Randomly adding double the number of nodes as edges in each layer
        u, v = np.random.choice(nodes_layer_p, 2, replace=False)
        G.add_edge(u, v, weight=1.0)

        u, v = np.random.choice(nodes_layer_t, 2, replace=False)
        G.add_edge(u, v, weight=1.0)

    # Add edges between corresponding nodes of different layers
    for i in range(num_nodes):
        G.add_edge(nodes_layer_p[i], nodes_layer_t[i], weight=inter_layer_weight)

    return G

###############################################################################
# %% OMICS GRAPH
def weighted_multi_omics_graph(cms, plot=False):
    p = 136
    kpa = 0
    cms = cms

    if "SLURM_JOB_ID" in os.environ:
        adj_matrix_proteomics = pd.read_csv(f'/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/inferred_adjacencies/proteomics_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv', index_col=0)
        adj_matrix_transcriptomics = pd.read_csv(f'/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/inferred_adjacencies/transcriptomics_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv', index_col=0)
    else: 
        adj_matrix_proteomics = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Networks/net_results/inferred_adjacencies/proteomics_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv', index_col=0)
        adj_matrix_transcriptomics = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Networks/net_results/inferred_adjacencies/transcriptomics_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv', index_col=0)


    # Create separate graphs for each adjacency matrix
    G_proteomics_layer = nx.from_pandas_adjacency(adj_matrix_proteomics)
    G_transcriptomic_layer = nx.from_pandas_adjacency(adj_matrix_transcriptomics)

    # Function to add a suffix to node names based on layer
    def add_layer_suffix(graph, suffix):
        return nx.relabel_nodes(graph, {node: f"{node}{suffix}" for node in graph.nodes})

    # Create separate graphs for each adjacency matrix and add layer suffix
    G_proteomics_layer = add_layer_suffix(nx.from_pandas_adjacency(adj_matrix_proteomics), '.p')
    G_transcriptomic_layer = add_layer_suffix(nx.from_pandas_adjacency(adj_matrix_transcriptomics), '.t')

    # Create a multiplex graph
    G_multiplex = nx.Graph()

    # Add nodes and edges from both layers
    G_multiplex.add_nodes_from(G_proteomics_layer.nodes(data=True), layer='PROT')
    G_multiplex.add_edges_from(G_proteomics_layer.edges(data=True), layer='PROT')
    G_multiplex.add_nodes_from(G_transcriptomic_layer.nodes(data=True), layer='RNA')
    G_multiplex.add_edges_from(G_transcriptomic_layer.edges(data=True), layer='RNA')

    common_nodes = set(adj_matrix_proteomics.index).intersection(adj_matrix_transcriptomics.index)

    inter_layer_weight = 1
    # Add nodes and edges from both layers
    G_multiplex.add_nodes_from(G_proteomics_layer.nodes(data=True), layer='PROT')
    G_multiplex.add_edges_from(G_proteomics_layer.edges(data=True), layer='PROT')
    G_multiplex.add_nodes_from(G_transcriptomic_layer.nodes(data=True), layer='RNA')
    G_multiplex.add_edges_from(G_transcriptomic_layer.edges(data=True), layer='RNA')

    common_nodes = set(adj_matrix_proteomics.index).intersection(adj_matrix_transcriptomics.index)

    inter_layer_weight = 1
    # Add inter-layer edges for common nodes
    for node in common_nodes:
        G_multiplex.add_edge(f"{node}.p", f"{node}.t",layer='interlayer')

    weighted_G_multiplex = G_multiplex.copy()
    for u, v, data in weighted_G_multiplex.edges(data=True):
        if data.get('layer') == 'interlayer':
            weighted_G_multiplex[u][v]['weight'] = inter_layer_weight
        else:
            weighted_G_multiplex[u][v]['weight'] = 1.0
    for u, v, data in weighted_G_multiplex.edges(data=True):
        if data.get('layer') == 'interlayer':
            weighted_G_multiplex[u][v]['weight'] = inter_layer_weight
        else:
            weighted_G_multiplex[u][v]['weight'] = 1.0

    ### VISUALIZE
    if not "SLURM_JOB_ID" in os.environ and plot:
        # Assume 'pos' is a dictionary of positions keyed by node
        pos = nx.spring_layout(weighted_G_multiplex)  # or any other layout algorithm

        # Shift proteomics nodes upward
        shift_amount = 0.5  # This is an arbitrary value for the amount of shift; you can adjust it as needed
        for node in G_proteomics_layer.nodes():
            pos[node][0] += shift_amount  # Shift the x-coordinate
            pos[node][1] += shift_amount  # Shift the y-coordinate


        # Now, draw the graph
        node_color_map = []
        for node in weighted_G_multiplex.nodes():
            if node.endswith('.p'):  # Proteomics nodes end with '.p'
                node_color_map.append('red')  # Color proteomics nodes red
            else:
                node_color_map.append('blue')  # Color other nodes blue

        # Draw nodes and edges separately to specify colors
        nx.draw_networkx_edges(weighted_G_multiplex, pos, alpha=0.4)
        nx.draw_networkx_nodes(weighted_G_multiplex, pos, node_color=node_color_map, alpha=0.8)
        nx.draw_networkx_labels(weighted_G_multiplex, pos, font_size=6, alpha=0.7)

        # Show the plot
        plt.show()

    return weighted_G_multiplex


# # Display some basic information about the multiplex graph
# if rank == 0:
#     num_nodes = G_multiplex.number_of_nodes()
#     num_edges = G_multiplex.number_of_edges()
#     num_nodes, num_edges

weighted_G_cms_123 = weighted_multi_omics_graph('cms123', plot=True)
weighted_G_cms_ALL = weighted_multi_omics_graph('cmsALL', plot=True)

# CHOOSING GRAPH #############################################################
# CHOOSING GRAPH #############################################################
# weighted_graph_use = weighted_G_cms_123
##############################################################################
# %%
################################################# ACTIVATE DOUBLE5 DEMO NET #########################################
# weighted_graph_use = double5_net




################################################# NODE AND DIFFUSION PARAMETERS  #########################################
# get hubs and low nodes
degree_dict = dict(weighted_G_cms_ALL.degree(weighted_G_cms_ALL.nodes()))
# get nodes with largest degree and smallest degree
hub_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:args.koh]
low_nodes = sorted(degree_dict, key=lambda x: degree_dict[x])[:args.kob]

if rank == 0:
    print(f'hub nodes: {hub_nodes}')
    print(f'anti-hubs nodes: {low_nodes}')

t_values = np.linspace(0.01, 10, 500)

# get args.red_range and convert to list
red_range = args.red_range.split(',')
# red_range = [float(i) for i in red_range]
red_range = [float(red_range[0]), float(red_range[1]), int(float(red_range[2]))]



nodes_to_investigate_bases = [node.split('.')[0] for node in hub_nodes + low_nodes] # FOR FIXED REDUCTION, NODE COMPARISON



# %% RUNS                                               ### MPI PARALLELIZATION ###
# Function to distribute nodes across ranks
def distribute_nodes(nodes, rank, size):
    num_nodes = len(nodes)
    nodes_per_proc = num_nodes // size
    remainder = num_nodes % size

    if rank < remainder:
        start_index = rank * (nodes_per_proc + 1)
        end_index = start_index + nodes_per_proc + 1
    else:
        start_index = remainder * (nodes_per_proc + 1) + (rank - remainder) * nodes_per_proc
        end_index = start_index + nodes_per_proc

    return nodes[start_index:end_index]


if "SLURM_JOB_ID" in os.environ:
    nodes_subset = distribute_nodes(nodes_to_investigate_bases, rank, size)
    print(f'nodes for rank {rank}: {nodes_subset}')
else:
    rank = 0
    size = 1
    nodes_subset = nodes_to_investigate_bases


# # Initialize containers for results
# orig_aggro_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values]
# orig_non_mesench_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_123), t) for t in t_values]
# orig_gdd_values = np.linalg.norm(np.array(orig_non_mesench_kernel) - np.array(orig_aggro_kernel), axis=(1, 2), ord='fro')

# local_results = {'original_gdds': orig_gdd_values, 'original_maxx_gdd': np.max(orig_gdd_values)}



# %%
# original_non_mesenchyme_G = copy.deepcopy(weighted_G_cms_123)
# weighted_lap_non_mesench = weighted_laplacian_matrix(original_non_mesenchyme_G)

# original_pan_G = copy.deepcopy(weighted_G_cms_ALL)
# weighted_lap_pan = weighted_laplacian_matrix(original_pan_G)

t_values = np.linspace(0.01, 10, 500)


# Create two multiplex graphs FOR TESTING
weighted_G_cms_123 = create_multiplex_graph(20)
weighted_G_cms_ALL = create_multiplex_graph(20)

# Example nodes subset
nodes_subset_with_suffix = list(weighted_G_cms_123.nodes())
nodes_subset = list(set([node.split('.')[0] for node in nodes_subset_with_suffix]))
print(f'nodes subset: {nodes_subset}')

# Initialize containers for results
orig_aggro_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values]
orig_non_mesench_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_123), t) for t in t_values]
orig_gdd_values = np.linalg.norm(np.array(orig_non_mesench_kernel) - np.array(orig_aggro_kernel), axis=(1, 2), ord='fro')
local_results = {'original_gdds': orig_gdd_values, 'original_maxx_gdd': np.max(orig_gdd_values)}
# print original max GDD
print(f'original max GDD: {np.max(orig_gdd_values)}')


# Initialize variables
knocked_out_nodes = []
global_min_gdds = [np.max(orig_gdd_values)]
stop_criteria_met = False
iteration_count = 0

# get the start time
start_time = time.time()

starting_max_gdd = np.max(orig_gdd_values)
reduction = 0.05

current_non_mesenchyme_G = copy.deepcopy(weighted_G_cms_123)
current_pan_G = copy.deepcopy(weighted_G_cms_ALL)

iteration_count = 0
while not stop_criteria_met and nodes_subset:
    print(f'iteration {iteration_count}')
    local_min_gdd = float('inf')
    local_best_node = None

    # Perform node knockout for each node in the subset
    for node_base in nodes_subset:
        local_results[node_base] = {}
        # NODE KNOCKOUT
        # Remove the node and recompute the Laplacian matrix
        knockdown_graph_aggro, knockdown_laplacian_aggro = knockdown_node_both_layers(weighted_G_cms_ALL, node_base, 
                                                                                        reduction_factor=reduction)

        knockdown_non_mesench, knockdown_laplacian_non_mesench = knockdown_node_both_layers(weighted_G_cms_123, node_base,
                                                                                        reduction_factor=reduction)

        # diff_kernel_non_mesench = [laplacian_exponential_kernel_eigendecomp(weighted_lap_non_mesench, t) for t in t_values]
        # diff_kernel_pan = [laplacian_exponential_kernel_eigendecomp(weighted_lap_pan, t) for t in t_values]
        diff_kernel_knock_aggro = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_aggro, t) for t in t_values]
        diff_kernel_knock_non_mesench = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_non_mesench, t) for t in t_values]

        # CALCULATE GDD
        # Compute the Frobenius norm of the difference between the kernels for each t
        gdd_values_trans = np.linalg.norm(np.array(diff_kernel_knock_non_mesench) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')

        max_gdd_trans = np.max(gdd_values_trans)

        # Update local best node if a lower max GDD is found
        if max_gdd_trans < local_min_gdd:
            local_min_gdd = max_gdd_trans
            local_best_node = node_base
            print(f'local best! {local_min_gdd}')

    # Gather results at the root rank
    all_min_gdds = comm.gather(local_min_gdd, root=0)
    all_best_nodes = comm.gather(local_best_node, root=0)

    # Root rank selects the global best node for knockout
    if rank == 0:
        global_min_gdd = min(all_min_gdds)
        global_best_node = all_best_nodes[all_min_gdds.index(global_min_gdd)]

        current_non_mesenchyme_G, _ = knockdown_node_both_layers(current_non_mesenchyme_G, global_best_node, reduction)
        current_pan_G, _ = knockdown_node_both_layers(current_pan_G, global_best_node, reduction)

        # Early stopping criterion
        if iteration_count > 0 and global_min_gdd > global_min_gdds[-1]:
            print(f'current min_gdd: {global_min_gdd}, previous: {global_min_gdds[-1]}')
            stop_criteria_met = True
        else:
            global_min_gdds.append(global_min_gdd)
            if rank == 0:
                knocked_out_nodes.append(global_best_node)

        # Prepare data for broadcasting (this may vary based on your graph structure)
        updated_graph_data = {
            'global_best_node': global_best_node,
            'reduction_factor': reduction,
        }

    # Broadcast decision to all ranks
    global_best_node = comm.bcast(global_best_node, root=0)
    stop_criteria_met = comm.bcast(stop_criteria_met, root=0)
    updated_graph_data = comm.bcast(updated_graph_data if rank == 0 else None, root=0)

    if rank != 0:
        # Apply the same knockout changes to the local graphs on each rank
        current_non_mesenchyme_G, _ = knockdown_node_both_layers(current_non_mesenchyme_G, updated_graph_data['global_best_node'], updated_graph_data['reduction_factor'])
        current_pan_G, _ = knockdown_node_both_layers(current_pan_G, updated_graph_data['global_best_node'], updated_graph_data['reduction_factor'])

    # Update nodes_subset and iteration count
    if not stop_criteria_met:
        nodes_subset.remove(global_best_node)
        iteration_count += 1
        nodes_subset = distribute_nodes(nodes_subset, rank, size)

# get the end time
end_time = time.time()
print(f'elapsed time (node knockdown calc) (rank {rank}): {end_time - start_time}')

all_results = comm.gather(local_results, root=0)

print(f'knocked out nodes: {knocked_out_nodes}')
print(f'global min gdds: {global_min_gdds}')

with open('diff_results/knocked_out_nodes.pkl', 'wb') as f:
        pkl.dump(knocked_out_nodes, f)
with open('diff_results/global_min_gdd.pkl', 'wb') as f:
    pkl.dump(global_min_gdd, f)

# # Post-processing on the root processor
# if rank == 0 and "SLURM_JOB_ID" in os.environ:
#     # Initialize a master dictionary to combine results
#     combined_results = {}

#     # Combine the results from each process
#     for process_results in all_results:
#         for key, value in process_results.items():
#             combined_results[key] = value

#     with open(f'diff_results/GDDs_and_Kernels_{str(diff_kernel_knock[0].shape[0])}.pkl', 'wb') as f:
#         pkl.dump(combined_results, f)
    
#     os.system("cp -r diff_results/ $HOME/thesis_code/Diffusion/")
#     print('Saving has finished.')


# else:
#     with open(f'diff_results/GDDs_and_Kernels_{str(diff_kernel_knock[0].shape[0])}.pkl', 'wb') as f:
#         pkl.dump(local_results, f)


MPI.Finalize()









# %% LOAD RESULTS
cms = 'cmsALL'

if not "SLURM_JOB_ID" in os.environ:
    with open(f'diff_results/GDDs_and_Kernels_268.pkl', 'rb') as f:
        GDDs_and_Kernels = pkl.load(f)

    print(f'GDDs_and_Kernels: {GDDs_and_Kernels.keys()}')
    print(f'Reduction factors: {GDDs_and_Kernels[list(GDDs_and_Kernels.keys())[0]].keys()}')

    # Choose the node and t_values for plotting
    selected_node = np.random.choice(list(GDDs_and_Kernels.keys()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

    # Left plot: GDD values over time for various reductions (single node)
    for reduction in GDDs_and_Kernels[selected_node].keys():
        gdd_values = GDDs_and_Kernels[selected_node][reduction]['gdd_values']
        ax1.plot(t_values, gdd_values, label=f'Reduction {reduction}')

    ax1.set_title(f'GDD Over Time for Various Reductions\nNode: {selected_node}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('GDD Value')
    ax1.legend()
    ax1.grid(True)

    # Choose a reduction factor from the list of reductions
    selected_reduction = red_range[1]

    max_gdds = {}
    # Right plot: GDD values over time for a single reduction (all nodes)
    for node_base in GDDs_and_Kernels.keys():
        gdd_values = GDDs_and_Kernels[node_base][selected_reduction]['gdd_values']
        ax2.plot(t_values, gdd_values, label=f'Node {node_base}', alpha=0.5)
        max_gdds[node_base] = np.max(gdd_values)

    ax2.set_title(f'GDD Over Time for Single Reduction\nReduction: {selected_reduction}')
    ax2.set_xlabel('Time')
    # ax2.set_ylabel('GDD Value')  # Y-label is shared with the left plot
    # ax2.legend()
    ax2.set_xlim([0, 2])
    ax2.grid(True)

    plt.show()
    # print(max_gdds)
    # max_GDD_1 = max_gdds['1']
    # max_GDD_2 = max_gdds['2']
    # print(max_GDD_1 - max_GDD_2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

    # PLOT 2 (WEAKER KNOCKDOWN)
    # make another plot as ax2 but for reduction factor = 0.8
    selected_reduction = red_range[-1]

    max_gdds = {}
    # Right plot: GDD values over time for a single reduction (all nodes)
    for node_base in GDDs_and_Kernels.keys():
        gdd_values = GDDs_and_Kernels[node_base][selected_reduction]['gdd_values']
        ax2.plot(t_values, gdd_values, label=f'Node {node_base}', alpha=0.5)
        max_gdds[node_base] = np.max(gdd_values)


    ax2.set_title(f'GDD Over Time for Single Reduction\nReduction: {selected_reduction}')
    ax2.set_xlabel('Time')
    # ax2.set_ylabel('GDD Value')  # Y-label is shared with the left plot
    # ax2.legend()
    ax2.set_xlim([0, 2])
    ax2.grid(True)

    plt.show()
    # print(max_gdds)
    # max_GDD_1 = max_gdds['1']
    # max_GDD_2 = max_gdds['2']
    # print(max_GDD_1 - max_GDD_2)

selected_reduction = red_range[-1]
# order nodes by max GDD
max_gdds = {}
for node_base in GDDs_and_Kernels.keys():
    max_gdds[node_base] = np.max(GDDs_and_Kernels[node_base][selected_reduction]['gdd_values'])

sorted_max_gdds = {k: v for k, v in sorted(max_gdds.items(), key=lambda item: item[1])}

# get the nodes with the highest GDD
highest_gdd_nodes = list(sorted_max_gdds.keys())[-5:]
highest_gdd_nodes



# %%
t_values = np.linspace(0.01, 10, 500)
red_args = '0.05,0.05,1'
red_range = red_args.split(',')
# red_range = [float(i) for i in red_range]
red_range = [float(red_range[0]), float(red_range[1]), int(float(red_range[2]))]

if not "SLURM_JOB_ID" in os.environ:
    with open(f'diff_results/GDDs_and_Kernels_268.pkl', 'rb') as f:
        GDDs_and_Kernels = pkl.load(f)

    print(f'GDDs_and_Kernels: {GDDs_and_Kernels.keys()}')
    print(f'Reduction factors: {GDDs_and_Kernels[list(GDDs_and_Kernels.keys())[0]].keys()}')
    # Choose a reduction factor from the list of reductions
    selected_reduction = red_range[1]

    # Calculate max GDDs for each node and sort them
    max_gdds_trans = {node: np.max(GDDs_and_Kernels[node][selected_reduction]['gdd_values_trans']) for node in GDDs_and_Kernels}
    max_gdds_disrupt = {node: np.max(GDDs_and_Kernels[node][selected_reduction]['gdd_values_disrupt']) for node in GDDs_and_Kernels}

    sorted_max_gdds_trans = sorted(max_gdds_trans.items(), key=lambda item: item[1])
    sorted_max_gdds_disrupt = sorted(max_gdds_disrupt.items(), key=lambda item: item[1])

    # Select top 3 and bottom 3 nodes for each case
    top_3_trans = [node for node, _ in sorted_max_gdds_trans[-3:]]
    bottom_3_trans = [node for node, _ in sorted_max_gdds_trans[:3]]
    top_3_disrupt = [node for node, _ in sorted_max_gdds_disrupt[-3:]]
    bottom_3_disrupt = [node for node, _ in sorted_max_gdds_disrupt[:3]]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=300)  # Creating 4 plots
    (ax1, ax2), (ax3, ax4) = axes

    # Plot GDD values for top 3 and bottom 3 nodes (Trans)
    for node in top_3_trans + bottom_3_trans:
        gdd_values_trans = GDDs_and_Kernels[node][selected_reduction]['gdd_values_trans']
        ax1.plot(t_values, gdd_values_trans, label=f'Node {node}')

    # Plot GDD values for top 3 and bottom 3 nodes (Disrupt)
    for node in top_3_disrupt + bottom_3_disrupt:
        gdd_values_disrupt = GDDs_and_Kernels[node][selected_reduction]['gdd_values_disrupt']
        ax3.plot(t_values, gdd_values_disrupt, label=f'Node {node}')

    ax1.set_title(f'GDD Over Time (Trans)\nTop 3 and Bottom 3 Nodes')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('GDD Value (Trans)')
    ax1.legend()
    ax1.grid(True)

    ax3.set_title(f'GDD Over Time (Disrupt)\nTop 3 and Bottom 3 Nodes')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('GDD Value (Disrupt)')
    ax3.legend()
    ax3.grid(True)

    plt.show()

# %%
# def separate_subgraph_layout(G, set1, set2, separation_vector):
#     """
#     Calculate a layout that visually separates two sets of nodes in a graph.

#     :param G: NetworkX graph
#     :param set1: First set of nodes
#     :param set2: Second set of nodes
#     :param separation_vector: A tuple (dx, dy) specifying how much to separate the layouts
#     :return: A dictionary of positions keyed by node
#     """
#     # Create subgraphs
#     subgraph1 = G.subgraph(set1)
#     subgraph2 = G.subgraph(set2)

#     # Compute layouts for subgraphs
#     layout1 = nx.spring_layout(subgraph1)
#     layout2 = nx.spring_layout(subgraph2)

#     # Shift the second layout
#     layout2 = {node: (pos[0] + separation_vector[0], pos[1] + separation_vector[1]) for node, pos in layout2.items()}

#     # Combine layouts
#     combined_layout = {**layout1, **layout2}
#     return combined_layout


# def plot_diffusion_process_for_two_graphs(graphs,  laplacians, set1, set2, times, node_to_isolate, start_node=9):
#     """
#     Plots the diffusion process on two graphs at specified times from a single starting node.

#     :param graphs: List of two NetworkX graphs
#     :param laplacians: List of two Laplacian matrices corresponding to the graphs
#     :param times: List of 3 times at which to visualize the diffusion
#     :param start_node: Node from which the diffusion starts
#     """
#     if len(graphs) != 2 or len(times) != 3:
#         print(len(graphs), len(times))
#         raise ValueError("Function requires exactly two graphs and three time points.")
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
#     fig.suptitle(f'Unperturbed graph (top), Knockdown of node {node_to_isolate} by factor {fixed_reduction} (bottom) s{rand_seed}', fontsize=25)


#     # layout=nx.spring_layout(graphs[0])
#     layout = separate_subgraph_layout(graphs[0], set1, set2, separation_vector=(4, 0))

#     label_layout = {node: (x - 0.1, y) for node, (x, y) in layout.items()}  # Shift labels to the left

#     for i, (G, L) in enumerate(zip(graphs, laplacians)):
#         for j, t in enumerate(times):
#             kernel = laplacian_exponential_diffusion_kernel(L, t)
#             heat_values = kernel[start_node, :]

#             if i == 1:  # If it's the second graph
#                 # get index of smallest heat value
#                 print(f'node with lowest heat == isolated node: {np.argmin(heat_values) == node_to_isolate}')
#                 sorted_heat_values = np.sort(heat_values)
#                 second_smallest_value = sorted_heat_values[1]  # Select the second smallest value
#                 norm = mcolors.Normalize(vmin=second_smallest_value, vmax=max(heat_values), clip=True)
#                 heat_values[node_to_isolate] = sorted_heat_values[0]
#             else:
#                 norm = mcolors.Normalize(vmin=min(heat_values), vmax=max(heat_values), clip=True)

#             # norm = mcolors.Normalize(vmin=min(heat_values), vmax=max(heat_values), clip=True)
#             mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)

#             nx.draw_networkx_edges(G, layout, alpha=0.2, ax=axes[i, j])
#             nx.draw_networkx_nodes(G, layout, node_size=200,
#                                    node_color=[mapper.to_rgba(value) for value in heat_values],
#                                    ax=axes[i, j])
#             nx.draw_networkx_labels(G, label_layout, font_size=20, ax=axes[i, j])
#             nx.draw_networkx_nodes(G, layout, nodelist=[node_to_isolate, start_node], node_size=300,
#                                node_color=['blue', 'red'],  # Example: red color
#                                ax=axes[i, j])
#             axes[i, j].set_title(f"Graph {i+1}, t={round(t, 2)}", fontsize=20)
#             axes[i, j].axis('off')

#     plt.colorbar(mapper, ax=axes[1, 2], shrink=0.7, aspect=20, pad=0.02)
#     plt.tight_layout()
#     plt.show()

# # Example usage
# fixed_reduction_index = np.where(np.isclose(red_range, fixed_reduction))[0][0]
# t_values = [0.1, max_gdd_times[fixed_reduction_index], 10]

# weighted_graph_use = copy.deepcopy(original_weighted_graph_use)
# weighted_lap_use = weighted_laplacian_matrix(weighted_graph_use)
# knockdown_graph, knockdown_laplacian = knockdown_node(weighted_graph_use, node_to_isolate, reduction_factor=fixed_reduction)

# seed_node = node_to_isolate + 1

# # Example usage with 3 graphs and their Laplacians for 9 different times
# plot_diffusion_process_for_two_graphs([weighted_graph_use, knockdown_graph], 
#                                            [weighted_lap_use, knockdown_laplacian], set_1, set_2,
#                                            t_values, start_node=seed_node, node_to_isolate=node_to_isolate)







# %% COMPARING DIRECT KERNEL WITH KERNEL EIGENDECOMPOSITION
# # start time
# start_time = time.time()
# # calculate laplacian_exponential_diffusion_kernel for knockdown graph for 5 different t values
# knockdown_kernel = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian, t) for t in t_values]

# mid_time = time.time()
# eigentimes = mid_time - start_time

# # check whether we get same result for laplacian_exponential_diffusion_kernel
# knockdown_kernel_direct = [laplacian_exponential_diffusion_kernel(knockdown_laplacian, t) for t in t_values]

# # end time
# end_time = time.time()
# directtimes = end_time - mid_time

# np.allclose(knockdown_kernel, knockdown_kernel_direct)

# print(f'Eigen-decomposition time: {eigentimes}')
# print(f'Direct computation time: {directtimes}')






### BELOW HERE IS THE CODE GRAVEYARD ##########################################



# # %% ############################ BARBELL GRAPHS ############################
# # Let's generate the three barbell graphs as shown in the image.
# # The first graph will be a complete barbell graph, the second will have its bridge removed, 
# # and the third will have its central connection removed.

# # Define the number of nodes in the barbell graph's complete subgraphs and the bridge length
# m1 = 5  # Number of nodes in the complete subgraphs
# m2 = 0  # Number of nodes in the bridge

# # Generate the complete barbell graph
# G_single_edge = nx.barbell_graph(m1, m2)

# # Identify the nodes to move and disconnect
# node_to_move_from_first_bell = m1 - 2  # Second to last node in the first bell
# node_to_move_from_second_bell = m1 + 1  # Second node in the second bell

# G_complete = G_single_edge.copy()
# # Add the new edge directly connecting the two identified nodes
# G_complete.add_edge(node_to_move_from_first_bell, node_to_move_from_second_bell)

# G_cc = G_complete.copy()
# # remove edge between nodes 7 and 9
# G_cc.remove_edge(7, 9)

# # Verify the graphs by plotting them
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# # Plot the complete barbell graph
# nx.draw(G_complete, ax=axes[0], with_labels=True)
# axes[0].set_title('Complete Barbell Graph')
# # Plot the barbell graph with the bridge removed
# nx.draw(G_single_edge, ax=axes[1], with_labels=True)
# axes[1].set_title('Barbell Graph with Bridge Removed')
# # Plot the barbell graph with the bell connection removed
# nx.draw(G_cc, ax=axes[2], with_labels=True)
# axes[2].set_title('Barbell Graph with bell connection removed')

# plt.tight_layout()
# plt.show()



# %% PLOTTING THE MAX GDD
# # Compute the Laplacian Matrices for both graphs
# laplacian_A = weighted_laplacian_matrix(G_complete)
# laplacian_B = weighted_laplacian_matrix(G_cc)

# # Step 3 and 4: Compute the Laplacian Kernels and calculate the xi values over a range of t values
# t_values = np.linspace(0, 10, 1000)
# xi_values = []

# for t in t_values:
#     kernel_A = laplacian_exponential_kernel_eigendecomp(*np.linalg.eigh(laplacian_A), t)
#     kernel_B = laplacian_exponential_kernel_eigendecomp(*np.linalg.eigh(laplacian_B), t)
#     xi = np.linalg.norm((kernel_A - kernel_B), 'fro')
#     xi_values.append(xi)

# # Find the maximum xi value and the corresponding t value using line search
# max_xi = max(xi_values)
# max_xi_index = xi_values.index(max_xi)
# max_xi_time = t_values[max_xi_index]

# # Step 5: Plot xi against t
# plt.figure(figsize=(10, 6))
# plt.plot(t_values, xi_values, label='xi(t)')
# plt.plot(max_xi_time, max_xi, 'ro', label='Max xi')
# # add a text label at maximum point
# plt.annotate(f'Max xi = {round(max_xi, 2)}\n at t = {round(max_xi_time, 2)}', xy=(max_xi_time, max_xi), xytext=(max_xi_time + 1, max_xi - 0.1),
#              )
# plt.xlabel('Diffusion Time (t)')
# plt.ylabel('Xi Value (Graph Difference)')
# plt.title('Xi Values Over Diffusion Time for Two Graphs')
# plt.legend()
# plt.grid(True)
# plt.show()


# %% TESTING EIGENDECOMP
# # TESTING ON SCALE_FREE GRAPH WITH RANDOM WEIGHTS
# weighted_laplacian = weighted_laplacian_matrix(weighted_scale_free_graph)

# # Eigen-decomposition of the Laplacian matrix
# eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
# # Test the eigen-decomposition approach for a single t value
# t_test = 0.5
# kernel_eigendecomp = laplacian_exponential_kernel_eigendecomp(eigenvalues, eigenvectors, t_test)
# # Compare with the direct computation for verification
# kernel_direct = laplacian_exponential_diffusion_kernel(laplacian_matrix, t_test)
# # Check if the results are similar (within a small numerical tolerance)
# np.allclose(kernel_eigendecomp, kernel_direct)

# # Test the eigen-decomposition approach with the weighted Laplacian matrix
# eigenvalues_weighted, eigenvectors_weighted = np.linalg.eigh(weighted_laplacian)
# kernel_eigendecomp_weighted = laplacian_exponential_kernel_eigendecomp(eigenvalues_weighted, eigenvectors_weighted, t_test)
# # Output the first few elements of the weighted Laplacian matrix and the kernel as a sanity check
# print(f'weighted laplacian:\n {weighted_laplacian[:5, :5]}')
# print(f'Reconstructed Kernel from eigen-decomposition (weighted):\n {kernel_eigendecomp_weighted[:5, :5]}')

# # np.allclose(kernel_eigendecomp, kernel_direct)

# %%
# %%
################################################################# GRAPH PARAMETERS
# N = 100  # Number of nodes
# m = 3    # Number of edges to attach from a new node to existing nodes



# # SCALE FREE GRAPH
# scale_free_graph = nx.barabasi_albert_graph(N, m, seed=rand_seed)
# laplacian_matrix = nx.laplacian_matrix(scale_free_graph).toarray()
# # Assign random weights to each edge (for example, weights between 0.1 and 1.0)
# weighted_scale_free_graph = scale_free_graph.copy()
# for u, v in weighted_scale_free_graph.edges():
#     weighted_scale_free_graph[u][v]['weight'] = np.random.uniform(0.1, 1.0)

# weighted_split_scalefree_g, set_1, set_2 = adjust_inter_set_edge_weights(weighted_scale_free_graph, new_weight=0.01)

# # get hub nodes
# degree_dict = dict(scale_free_graph.degree(scale_free_graph.nodes()))
# # get 3 nodes with largest degree
# hub_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:3]
# low_nodes = sorted(degree_dict, key=lambda x: degree_dict[x])[:3]
# print(f'hub nodes: {hub_nodes}')
# print(f'anti-hubs nodes: {low_nodes}')


# # RANDOM GRAPH
# random_graph = nx.erdos_renyi_graph(N, 0.5, seed=rand_seed) 
# random_laplacian = nx.laplacian_matrix(random_graph).toarray()
# weighted_random_graph = random_graph.copy()
# for u, v in weighted_random_graph.edges():
#     weighted_random_graph[u][v]['weight'] = 1.0


# %%
# %%
################################################################# GRAPH PARAMETERS
# N = 100  # Number of nodes
# m = 3    # Number of edges to attach from a new node to existing nodes



# # SCALE FREE GRAPH
# scale_free_graph = nx.barabasi_albert_graph(N, m, seed=rand_seed)
# laplacian_matrix = nx.laplacian_matrix(scale_free_graph).toarray()
# # Assign random weights to each edge (for example, weights between 0.1 and 1.0)
# weighted_scale_free_graph = scale_free_graph.copy()
# for u, v in weighted_scale_free_graph.edges():
#     weighted_scale_free_graph[u][v]['weight'] = np.random.uniform(0.1, 1.0)

# weighted_split_scalefree_g, set_1, set_2 = adjust_inter_set_edge_weights(weighted_scale_free_graph, new_weight=0.01)

# # get hub nodes
# degree_dict = dict(scale_free_graph.degree(scale_free_graph.nodes()))
# # get 3 nodes with largest degree
# hub_nodes = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:3]
# low_nodes = sorted(degree_dict, key=lambda x: degree_dict[x])[:3]
# print(f'hub nodes: {hub_nodes}')
# print(f'anti-hubs nodes: {low_nodes}')


# # RANDOM GRAPH
# random_graph = nx.erdos_renyi_graph(N, 0.5, seed=rand_seed) 
# random_laplacian = nx.laplacian_matrix(random_graph).toarray()
# weighted_random_graph = random_graph.copy()
# for u, v in weighted_random_graph.edges():
#     weighted_random_graph[u][v]['weight'] = 1.0
