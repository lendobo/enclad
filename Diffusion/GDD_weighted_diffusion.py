# %%
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
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
import csv
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
import pymnet as pn

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
parser.add_argument('--koh', type=int, default=40, help='Number of hub nodes to knock out')
parser.add_argument('--kob', type=int, default=5, help='Number of bottom nodes to knock out')
parser.add_argument('--red_range', type=str, default='0.05,0.9,3', help='Range of reduction factors to investigate')
parser.add_argument('--cms', type=str, default='cmsALL', choices=['cmsALL', 'cms123'], help='CMS to use')
# parser.add_argument('--mode', type=str, default='disruption', choices=['disruption', 'transition'], help='Type of knockout analysis')
parser.add_argument('--pathway', type=bool, default=False, help='Boolean for Pathway Knockout')
parser.add_argument('--test_net', type=bool, default=False, help='Boolean for testing network')
parser.add_argument('--permu_runs', type=int, default=30, help='Number of runs for permutation random pathway knockout')
parser.add_argument('--visualize', type=bool, default=False, help='Boolean for visualizing the network')
parser.add_argument('--enrich_file', type=str, default='/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/Pathway_Enrichment_Info.csv', help='Path to pathway enrichment file')

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
    node_order = list(G.nodes())
    # Weighted adjacency matrix
    W = nx.to_numpy_array(G, nodelist=node_order, weight='weight')
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

def knockdown_node_both_layers(G, node_to_isolate_base, reduced_weight=0.3):
    """
    Reduces the weights of all edges connected to a node in both layers of the graph.

    :param G: NetworkX graph
    :param node_to_isolate_base: Base node name whose edges will be reduced in both layers
    :param node_to_isolate_base: Base node name whose edges will be reduced in both layers
    :param reduced_weight: Factor to reduce edge weights by, defaults to 0.5
    :return: Tuple containing the modified graph and its weighted Laplacian matrix
    """

    modified_graph = G.copy()
    
    # Add layer suffixes to the base node name
    node_to_isolate_proteomics = f"{node_to_isolate_base}.p"
    node_to_isolate_transcriptomics = f"{node_to_isolate_base}.t"
    
    # Reduce the weight of all edges to and from this node in both layers
    for node_to_isolate in [node_to_isolate_proteomics, node_to_isolate_transcriptomics]:
        for neighbor in G[node_to_isolate]:
            modified_graph[node_to_isolate][neighbor]['weight'] = reduced_weight
            modified_graph[neighbor][node_to_isolate]['weight'] = reduced_weight
    
    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = weighted_laplacian_matrix(modified_graph)
    return modified_graph, new_laplacian

def knockdown_pathway_nodes(G, pathway_description, reduced_weight=0.3):
    """
    Reduces the weights of all edges connected to the nodes in a pathway in both layers of the graph.

    :param G: NetworkX graph
    :param pathway_description: Description of the pathway whose nodes will be reduced in both layers
    :param reduced_weight: Factor to reduce edge weights by, defaults to 0.3
    :return: Tuple containing the modified graph and its weighted Laplacian matrix
    """

    # Find rows where 'description' column contains the given string
    rows = pathway_df[pathway_df['description'].str.contains(pathway_description)]
    
    # Initialize a list to store the base node names
    base_node_names = []

    # Iterate over the found rows
    for _, row in rows.iterrows():
        # Split the 'genes' column into individual genes and add them to the list
        base_node_names.extend(row['genes'].split('|'))

    modified_graph = G.copy()
    
    # Iterate over the base node names
    for node_to_isolate_base in base_node_names:
        # Add layer suffixes to the base node name
        node_to_isolate_proteomics = f"{node_to_isolate_base}.p"
        node_to_isolate_transcriptomics = f"{node_to_isolate_base}.t"
        
        # Reduce the weight of all edges to and from this node in both layers
        for node_to_isolate in [node_to_isolate_proteomics, node_to_isolate_transcriptomics]:
            if node_to_isolate in G:
                for neighbor in G[node_to_isolate]:
                    modified_graph[node_to_isolate][neighbor]['weight'] = reduced_weight
                    modified_graph[neighbor][node_to_isolate]['weight'] = reduced_weight
    
    # Compute the weighted Laplacian matrix for the modified graph
    new_laplacian = weighted_laplacian_matrix(modified_graph)

    return modified_graph, new_laplacian


def knockdown_random_nodes(G, nodes, reduced_weight=0.3):
    """
    Reduces the weights of all edges connected to the nodes in a pathway or a list of nodes in both layers of the graph.

    :param G: NetworkX graph
    :param nodes: List of nodes whose edges will be reduced in both layers
    :param reduced_weight: Factor to reduce edge weights by, defaults to 0.3
    :return: Tuple containing the modified graph and its weighted Laplacian matrix
    """
    modified_graph = G.copy()
    
    # Iterate over the node names
    for node_to_isolate_base in nodes:
        # Add layer suffixes to the base node name
        node_to_isolate_proteomics = f"{node_to_isolate_base}.p"
        node_to_isolate_transcriptomics = f"{node_to_isolate_base}.t"
        
        # Reduce the weight of all edges to and from this node in both layers
        for node_to_isolate in [node_to_isolate_proteomics, node_to_isolate_transcriptomics]:
            if node_to_isolate in G:
                for neighbor in G[node_to_isolate]:
                    modified_graph[node_to_isolate][neighbor]['weight'] = reduced_weight
                    modified_graph[neighbor][node_to_isolate]['weight'] = reduced_weight
    
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

def create_multiplex_test(num_nodes, inter_layer_weight=1.0):
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
        adj_matrix_proteomics = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/MONIKA/Networks/net_results/inferred_adjacencies/proteomics_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv', index_col=0)
        adj_matrix_transcriptomics = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/MONIKA/Networks/net_results/inferred_adjacencies/transcriptomics_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv', index_col=0)


    # Create separate graphs for each adjacency matrix
    G_proteomics_layer = nx.from_pandas_adjacency(adj_matrix_proteomics)
    G_transcriptomic_layer = nx.from_pandas_adjacency(adj_matrix_transcriptomics)

    if plot:
        # Calculate the degrees of each node
        degrees = [degree for node, degree in G_proteomics_layer.degree()]
        # Sort the degrees
        degrees_sorted = sorted(degrees, reverse=True)
        # Create an array representing the index of each degree for x-axis
        x = range(len(degrees_sorted))
        # Plotting the line chart
        plt.plot(x, degrees_sorted, label='Transcriptomics Degrees')
        # Adding labels and legend
        plt.legend()
        plt.xlabel('Index')
        plt.ylabel('Degree')
        plt.title('Ordered Degree Distribution')
        plt.show()

        # make a histogram of the degrees
        plt.hist(degrees, bins=20)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.show()


    # # OR TRY ON A RANDOM ER GRAPH
    # G_proteomics_layer = nx.erdos_renyi_graph(100, 0.1)

    # Function to add a suffix to node names based on layer
    def add_layer_suffix(graph, suffix):
        return nx.relabel_nodes(graph, {node: f"{node}{suffix}" for node in graph.nodes})

    # Create separate graphs for each adjacency matrix and add layer suffix
    G_proteomics_layer = add_layer_suffix(nx.from_pandas_adjacency(adj_matrix_proteomics), '.p')
    G_transcriptomic_layer = add_layer_suffix(nx.from_pandas_adjacency(adj_matrix_transcriptomics), '.t')

    # Create a multiplex graph
    G_multiplex = nx.Graph()

    # Add nodes and edges from both layers
    G_multiplex.add_nodes_from(G_proteomics_layer.nodes(data=True), layer='PROTEIN')
    G_multiplex.add_edges_from(G_proteomics_layer.edges(data=True), layer='PROTEIN')
    G_multiplex.add_nodes_from(G_transcriptomic_layer.nodes(data=True), layer='RNA')
    G_multiplex.add_edges_from(G_transcriptomic_layer.edges(data=True), layer='RNA')

    common_nodes = set(adj_matrix_proteomics.index).intersection(adj_matrix_transcriptomics.index)

    inter_layer_weight = 1
    # Add nodes and edges from both layers
    G_multiplex.add_nodes_from(G_proteomics_layer.nodes(data=True), layer='PROTEIN')
    G_multiplex.add_edges_from(G_proteomics_layer.edges(data=True), layer='PROTEIN')
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


    # PYMNET BS
    # Initialize a pymnet multilayer network
    M = pn.MultiplexNetwork(couplings=('categorical', 1), fullyInterconnected=False)

    def preprocess_node_name(node_name):
        # If node ends with '.p' or '.t', remove the suffix
        if node_name.endswith('.p') or node_name.endswith('.t'):
            return node_name[:-2]  # Assuming suffixes are always two characters
        return node_name

    # Add nodes and edges for proteomics layer
    for node in G_proteomics_layer.nodes:
        # Preprocess node names to remove suffixes
        processed_node = preprocess_node_name(node)
        M.add_node(processed_node, layer='PROTEIN')
    for u, v in G_proteomics_layer.edges:
        # Preprocess node names for each edge
        processed_u = preprocess_node_name(u)
        processed_v = preprocess_node_name(v)
        M[processed_u, processed_v, 'PROTEIN', 'PROTEIN'] = 1

    # Add nodes and edges for transcriptomic layer
    for node in G_transcriptomic_layer.nodes:
        # Preprocess node names to remove suffixes
        processed_node = preprocess_node_name(node)
        M.add_node(processed_node, layer='RNA')
    for u, v in G_transcriptomic_layer.edges:
        # Preprocess node names for each edge
        processed_u = preprocess_node_name(u)
        processed_v = preprocess_node_name(v)
        M[processed_u, processed_v, 'RNA', 'RNA'] = 1

    return weighted_G_multiplex, M  #, rna_node_positions       



# # Display some basic information about the multiplex graph
# if rank == 0:
#     num_nodes = G_multiplex.number_of_nodes()
#     num_edges = G_multiplex.number_of_edges()
#     num_nodes, num_edges

weighted_G_cms_123, pymnet_123 = weighted_multi_omics_graph('cms123', plot=True)
weighted_G_cms_ALL, pymnet_ALL = weighted_multi_omics_graph('cmsALL', plot=False)


# CHOOSING GRAPH #############################################################
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
# red_range = [float(i) for i in red_range]
red_range = args.red_range.split(',')
red_range = np.linspace(float(red_range[0]), float(red_range[1]), int(float(red_range[2])))


if args.koh == 0:
    nodes_to_investigate_bases = [node.split('.')[0] for node in weighted_G_cms_ALL.nodes()] # FOR FIXED REDUCTION, NODE COMPARISON
else:
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


# Function to distribute pathways across ranks
def distribute_pathways(pathways, rank, size):
    num_pathways = len(pathways)
    pathways_per_proc = num_pathways // size
    remainder = num_pathways % size

    if rank < remainder:
        start_index = rank * (pathways_per_proc + 1)
        end_index = start_index + pathways_per_proc + 1
    else:
        start_index = remainder * (pathways_per_proc + 1) + (rank - remainder) * pathways_per_proc
        end_index = start_index + pathways_per_proc

    return pathways[start_index:end_index]


def run_knockout_analysis(G_aggro, G_stable, knockout_type, knockout_target, red_range, t_values, orig_aggro_kernel, orig_gdd_values, pathway_df=None, num_runs=30):
    results = {}

    if knockout_type == 'runtype_node' or knockout_type == 'runtype_pathway':

        if knockout_type == 'runtype_pathway':
            target_list = pathway_df[pathway_df['description'].str.contains(knockout_target)]['genes'].str.split('|').explode().tolist()
            num_genes = len(target_list)  # Number of genes in the pathway
            print(f"Gene count in {knockout_target}: {num_genes}\n")


        results[knockout_target] = {}
        for reduction in red_range:
            print(f"Processing {knockout_target} Knockdown with reduction factor: {reduction}")
            # Perform the knockout
            knockdown_func = knockdown_node_both_layers if knockout_type == 'runtype_node' else knockdown_pathway_nodes
            knockdown_graph_aggro, knockdown_laplacian_aggro = knockdown_func(G_aggro, knockout_target, reduced_weight=reduction)
            knockdown_non_mesench, knockdown_laplacian_non_mesench = knockdown_func(G_stable, knockout_target, reduced_weight=reduction)

            # Calculate diffusion kernels and GDD
            diff_kernel_knock_aggro = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_aggro, t) for t in t_values]
            diff_kernel_knock_non_mesench = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_non_mesench, t) for t in t_values]

            gdd_values_trans = np.linalg.norm(np.array(diff_kernel_knock_non_mesench) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')
            gdd_values_disrupt = np.linalg.norm(np.array(orig_aggro_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')

            results[knockout_target][reduction] = {
                'gdd_values_trans': gdd_values_trans,
                'gdd_values_disrupt': gdd_values_disrupt,
                'max_gdd_trans': np.max(gdd_values_trans),
                'max_gdd_disrupt': np.max(gdd_values_disrupt)
            }

            if args and args.visualize:
                results[knockout_target][reduction]['vis_kernels'] = [diff_kernel_knock_aggro[i] for i, t in enumerate(t_values) if i % 20 == 0]

    elif knockout_type == 'runtype_random':
        all_nodes = list(set([node.split('.')[0] for node in G_aggro.nodes()]))

        for _ in range(num_runs):
            random.seed(_)
            random_nodes = random.sample(all_nodes, knockout_target)
            max_gdd_trans_run, max_gdd_disrupt_run = [], []

            results[f'random_{knockout_target}_run_{_}'] = {}
            for reduction in red_range:
                # print(f"Random Pathway Knockout: size {knockout_target}, run {_} with reduction factor: {reduction}")
                # Perform the knockout
                knockdown_graph_aggro, knockdown_laplacian_aggro = knockdown_random_nodes(G_aggro, random_nodes, reduced_weight=reduction)
                knockdown_non_mesench, knockdown_laplacian_non_mesench = knockdown_random_nodes(G_stable, random_nodes, reduced_weight=reduction)

                # Calculate diffusion kernels and GDD
                diff_kernel_knock_aggro = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_aggro, t) for t in t_values]
                diff_kernel_knock_non_mesench = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_non_mesench, t) for t in t_values]

                gdd_values_trans = np.linalg.norm(np.array(diff_kernel_knock_non_mesench) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')
                gdd_values_disrupt = np.linalg.norm(np.array(orig_aggro_kernel) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')

                max_gdd_trans_run.append(np.max(gdd_values_trans))
                max_gdd_disrupt_run.append(np.max(gdd_values_disrupt))

                results[f'random_{knockout_target}_run_{_}'][reduction] = {
                    'max_gdd_trans': max(max_gdd_trans_run),
                    'max_gdd_disrupt': max(max_gdd_disrupt_run)
                }

    return results


# %%
# Joy's Pathways
file_path = '/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/MONIKA/data/Joy_enrichment/results data folder attempt 281223/trans/gem/gProfiler_hsapiens_28-12-2023_11-20-54 am.gem.txt'

joy_path_df = pd.read_csv(file_path, sep='\t')
joy_path_df.head()

# %%


crit_paths = ['Angiogenesis', 'Regulation of angiogenesis', 'Positive regulation of angiogenesis', 'Sprouting angiogenesis', 
                      'Regulation of cell migration involved in sprouting angiogenesis', 'TGF-beta signaling pathway', 'TGF-beta receptor signaling'
                      'TGF-beta signaling in thyroid cells for epithelial-mesenchymal transition', 'Wnt signaling pathway and pluripotency',
                      'Signaling by TGFB family members', 'Canonical and non-canonical TGF-B signaling', 'Transforming growth factor beta receptor signaling pathway',
                      'Cellular response to transforming growth factor beta stimulus', 'Regulation of transforming growth factor beta2 production',
                      'Regulation of transforming growth factor beta receptor signaling pathway', 'Negative regulation of transforming growth factor beta receptor signaling pathway']
pathways = crit_paths

args.pathway = True
args.enrich_file = '/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/MONIKA/Diffusion/data/Pathway_Enrichment_Info.csv'

if args.pathway:
    pathway_df = pd.read_csv(args.enrich_file) # '/home/mbarylli/thesis_code/Diffusion/data_for_diffusion/Pathway_Enrichment_Info.csv'
    pathway_df = pathway_df.drop_duplicates(subset='description', keep='first')


    # # only keep first X
    # filtered_pathway_df = filtered_pathway_df.head(args.koh)
    # # Start with critical pathways

    # check overlap between 'Description' in joy_path_df and 'description' in pathway_df
    filtered_pathway_df = pathway_df[pathway_df['description'].isin(joy_path_df['Description'])]
    # keep only rows which have a perc overlap above 0.5
    filtered_pathway_df = filtered_pathway_df[filtered_pathway_df['perc overlap'] >= 0.5]
    # Filter the dataframe to include only pathways with '# genes' between 1 and 25
    filtered_pathway_df = filtered_pathway_df[(filtered_pathway_df['# genes'] >= 5) & (filtered_pathway_df['# genes'] <= 25)]
    print(filtered_pathway_df.shape)

    # Add unique pathways from the filtered list until you reach args.koh
    for pathway in filtered_pathway_df['description'].tolist():
        if len(pathways) >= args.koh:
            break
        if pathway not in pathways:
            pathways.append(pathway)

    # random pathway length distribution
    matches = pathway_df['description'].isin(pathways)
    interest_pathway_df = pathway_df[matches]
    pathway_lengths = [len(row['genes'].split('|')) for _, row in interest_pathway_df.iterrows()]

    if "SLURM_JOB_ID" in os.environ:
        # Distribute pathways across ranks
        pathways_subset = distribute_pathways(pathways, rank, size)
        rand_lengths_subset = distribute_pathways(pathway_lengths, rank, size)
        
        print(f'pathways for rank {rank}: {pathways_subset}')
        print(f'random pathway size for rank {rank}: {rand_lengths_subset}')
    
        
    else: #Otherwise, if run locally
        pathways_subset = pathways
        rand_lengths_subset = pathway_lengths

        print(f'pathways for rank {rank}: {pathways_subset}')
        print(f'rand pathways for rank {rank}: {rand_lengths_subset}')


# DISTRIBUTE NODES ACROSS RANKS
if "SLURM_JOB_ID" in os.environ:
    # Distribute nodes across ranks
    nodes_subset = distribute_nodes(nodes_to_investigate_bases, rank, size)
    print(f'nodes for rank {rank}: {nodes_subset}')
else:
    nodes_subset = nodes_to_investigate_bases
    print(f'nodes for rank {rank}: {nodes_subset}')
    rank = 0
    size = 1


# RUN on test net
if args.test_net:
    t_values = np.linspace(0.01, 10, 500)
    # Create two multiplex graphs FOR TESTING
    weighted_G_cms_123 = create_multiplex_test(12)
    weighted_G_cms_ALL = create_multiplex_test(12)

    # Example nodes subset
    nodes_subset_with_suffix = list(weighted_G_cms_123.nodes())
    nodes_subset = list(set([node.split('.')[0] for node in nodes_subset_with_suffix]))
    print(f'nodes subset: {nodes_subset}')

    # Initialize containers for results
    orig_aggro_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values]
    orig_non_mesench_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_123), t) for t in t_values]
    orig_gdd_values = np.linalg.norm(np.array(orig_non_mesench_kernel) - np.array(orig_aggro_kernel), axis=(1, 2), ord='fro')
else:
    # Initialize containers for results
    orig_aggro_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_ALL), t) for t in t_values]
    orig_non_mesench_kernel = [laplacian_exponential_kernel_eigendecomp(weighted_laplacian_matrix(weighted_G_cms_123), t) for t in t_values]
    orig_gdd_values = np.linalg.norm(np.array(orig_non_mesench_kernel) - np.array(orig_aggro_kernel), axis=(1, 2), ord='fro')


# get max orig_gdd_values
max_orig_gdd_values = np.max(orig_gdd_values)























# %% 
# get the start time
start_time = time.time()

local_target_results = {}

if args.pathway:
    # PATHWAY KNOCKOUTS
    for pathway in pathways_subset:
        pathway_results = run_knockout_analysis(weighted_G_cms_ALL, weighted_G_cms_123, 'runtype_pathway', pathway, red_range, t_values, orig_aggro_kernel, orig_gdd_values, pathway_df)
        local_target_results.update(pathway_results)

    # RANDOM PATHWAY KNOCKOUTS (for permutation analysis)
    local_rand_results = {}
    # Create a boolean series where each element is True if the 'description' column contains any of the pathway descriptions
    for random_pathway_size in rand_lengths_subset:
        rand_results = run_knockout_analysis(weighted_G_cms_ALL, weighted_G_cms_123, 'runtype_random', random_pathway_size, red_range, t_values, orig_aggro_kernel, orig_gdd_values, pathway_df, num_runs=args.permu_runs)
        local_rand_results.update(rand_results)
        
    print(f'Rank {rank} has finished the target and random pathway runs.')

else:
    for node in nodes_subset:
        # TESTING the knockout analysis function
        node_results = run_knockout_analysis(weighted_G_cms_ALL, weighted_G_cms_123, 'runtype_node', node, red_range, t_values, orig_aggro_kernel, orig_gdd_values)
        local_target_results.update(node_results)


# GATHERING RESULTS
all_target_results = comm.gather(local_target_results, root=0)
if args.pathway:
    all_rand_results = comm.gather(local_rand_results, root=0)
    all_results_list = [all_target_results, all_rand_results]
    filename_identifiers = ['target', 'random']
    
    # Assuming 'pathways' is a list of strings
    unique_identifier = ''.join([pathway[0] for pathway in pathways])
    unique_identifier = unique_identifier[:20]


else:
    all_results_list = [all_target_results]
    filename_identifiers = ['target']

    unique_identifier = ''.join([node[0] for node in nodes_to_investigate_bases])
    unique_identifier = unique_identifier[:20]

for i, all_results in enumerate(all_results_list):
    # Post-processing on the root processor
    if rank == 0 and "SLURM_JOB_ID" in os.environ:
        # Initialize a master dictionary to combine results
        combined_results = {}

        # Combine the results from each process
        for process_results in all_results:
            for key, value in process_results.items():
                combined_results[key] = value

        with open(f'diff_results/Pathway_{args.pathway}_{filename_identifiers[i]}_{unique_identifier}_GDDs_ks{str(orig_aggro_kernel[0].shape[0])}.pkl', 'wb') as f:
            pkl.dump(combined_results, f)
        
        os.system("cp -r diff_results/ $HOME/thesis_code/Diffusion/")
        print('Saving has finished.')


    elif rank == 0 and "SLURM_JOB_ID" not in os.environ:
        with open(f'diff_results/Pathway_{args.pathway}_{filename_identifiers[i]}_{unique_identifier}_GDDs_ks{str(orig_aggro_kernel[0].shape[0])}.pkl', 'wb') as f:
            pkl.dump(local_target_results, f)

        # with open(f'diff_results/Pathway_{args.pathway}_random_node_{unique_identifier}_GDDs_ks{str(orig_aggro_kernel[0].shape[0])}.pkl', 'wb') as f:
        #     pkl.dump(local_rand_results, f)


# get the end time
end_time = time.time()
print(f'elapsed time (node knockdown calc) (rank {rank}): {end_time - start_time}')



MPI.Finalize()




# %% SIGNIFICANCE TESTING

if args.pathway:
    # Load results and pathway information
    with open('diff_results/Pathway_True_target_ARPSRTTWSCTCRRNLRPHC_GDDs_ks272.pkl', 'rb') as f:
        target_results = pkl.load(f)
    with open('diff_results/Pathway_True_random_ARPSRTTWSCTCRRNLRPHC_GDDs_ks272.pkl', 'rb') as f:
        random_results = pkl.load(f)

    def get_pathway_length(pathway_name, df):
        if pathway_name in df['description'].values:
            row = df[df['description'] == pathway_name].iloc[0]
            return len(row['genes'].split('|'))
        else:
            # Handle the case where pathway_name is not found in df
            # For example, return a default value or raise a custom error
            return None  # or raise ValueError(f"Pathway '{pathway_name}' not found in dataframe")

    # Function to parse random results keys
    def parse_random_key(key):
        parts = key.split('_')
        return int(parts[1]), int(parts[3])  # length, run_number

    # Organize random results by pathway length
    random_results_by_length = {}  # {pathway_length: [list of max_gdd_trans values]}
    for key, result in random_results.items():
        length, run_number = parse_random_key(key)
        max_gdd_trans = result[0.05]['max_gdd_trans']
        if length not in random_results_by_length:
            random_results_by_length[length] = []
        random_results_by_length[length].append(max_gdd_trans)

    # Calculate p-values
    p_values = {}
    for pathway, result in target_results.items():
        target_max_gdd_trans = result[0.05]['max_gdd_trans']
        pathway_length = get_pathway_length(pathway, interest_pathway_df)
        if pathway_length is not None:
            random_distribution = random_results_by_length.get(pathway_length, [])
            if random_distribution:  # Ensure there are random results for this length
                p_value = percentileofscore(random_distribution, target_max_gdd_trans, kind='mean') / 100
                # if p_value == 0:
                #     p_value = "<0.001"
                p_values[pathway] = p_value
            else:
                # Handle case where there are no random results for this length
                p_values[pathway] = None  # or some other placeholder
        else:
            print(f"Pathway length not found for {pathway}")
            # Handle case where pathway length could not be determined
            p_values[pathway] = None  # or some other placeholder


    # Filter out None values from p_values
    filtered_p_values = {k: v for k, v in p_values.items() if v is not None}

    # Adjust for multiple testing on the filtered p-values
    adjusted_p_values = multipletests(list(filtered_p_values.values()), method='fdr_bh')[1]
    adjusted_p_values_dict = dict(zip(filtered_p_values.keys(), adjusted_p_values))

    # Merge adjusted p-values back with the original set (assigning None where appropriate)
    final_adjusted_p_values = {pathway: adjusted_p_values_dict.get(pathway, None) for pathway in p_values.keys()}

    # Interpret results
    significant_pathways = {pathway: adj_p for pathway, adj_p in final_adjusted_p_values.items() if adj_p is not None and adj_p <= 0.05}

    # Display significant pathways
    print("Significant Pathways (adjusted p-value ≤ 0.05):")
    for pathway, adj_p in significant_pathways.items():
        print(f"{pathway}: {adj_p}")


    # Create a DataFrame for target results including pathway length
    data = []

    # put max_orig_gdd_values in the first row
    data.append(['ORIGINAL_MAX_GDD', max_orig_gdd_values, None, None])

    for pathway, result in target_results.items():
        max_gdd_trans = result[0.05]['max_gdd_trans']
        p_value = p_values.get(pathway, None)  # Get the p-value, if available
        pathway_length = get_pathway_length(pathway, interest_pathway_df)  # Get pathway length
        data.append([pathway, max_gdd_trans, p_value, pathway_length])

    # Creating the DataFrame with an additional column for pathway length
    target_df = pd.DataFrame(data, columns=['Pathway', 'Max_GDD_Trans', 'P_Value', 'Pathway_Length'])


    # Sorting the DataFrame by Max_GDD_Trans in ascending order
    target_df_sorted = target_df.sort_values(by='P_Value', ascending=True)

    # Display the first few rows of the sorted DataFrame
    print(target_df_sorted.head())

    # write to csv
    target_df_sorted.to_csv(f'diff_results/Pathway_Knockouts_{unique_identifier}_permu_{args.permu_runs}.csv', index=False)


    # %%
    # Filter for pathways with 0 p-value
    zero_p_value_pathways = [pathway for pathway, p_val in p_values.items() if p_val != 0]

    # Plot distributions
    for pathway in zero_p_value_pathways:
        # Get target max_gdd_trans value
        target_max_gdd_trans = target_results[pathway][0.05]['max_gdd_trans']

        # Get pathway length
        pathway_length = get_pathway_length(pathway, interest_pathway_df)

        # Get corresponding random distribution
        random_distribution = random_results_by_length.get(pathway_length, [])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.hist(random_distribution, bins=30, alpha=0.7, label='Random Distribution')
        plt.axvline(x=target_max_gdd_trans, color='r', linestyle='dashed', linewidth=2, label='Target Value')
        plt.title(f"Distribution for Pathway: {pathway} (Length: {pathway_length})")
        plt.xlabel('Max GDD Trans')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()



# %%
# VISUALIZE DIFFUSION

def multiplex_net_viz(M, ax, diff_colors=False, node_colors=None, node_sizes=None, node_coords=None):

    dark_red = "#8B0000"  # Dark red color
    dark_blue = "#00008B"  # Dark blue color

    # Iterate over nodes in the 'PROTEIN' layer
    # Initialize a dictionary to hold node colors
    for node in M.iter_nodes(layer='PROTEIN'):
        if not diff_colors:
            node_colors[(node, 'PROTEIN')] = dark_red

    # Iterate over nodes in the 'RNA' layer
    for node in M.iter_nodes(layer='RNA'):
        if not diff_colors:
            node_colors[(node, 'RNA')] = dark_blue


    edge_color = "#505050"  # A shade of gray, for example

    # Initialize a dictionary to hold edge colors
    edge_colors = {}

    # Assign colors to edges in the 'PROTEIN' and 'RNA' layers
    # Assuming edges are between nodes within the same layer
        
    layer_colors = {'PROTEIN': "red", 'RNA': "blue"}

    fig = pn.draw(net=M,
            ax=ax,
            show=False, 
            nodeColorDict=node_colors,
            nodeLabelDict={nl: None for nl in M.iter_node_layers()},  # Set all node labels to None explicitly
            nodeLabelRule={},  # Clear any label rules
            defaultNodeLabel=None,  # Set default label to None
            nodeSizeDict=node_sizes,
            # nodeSizeRule={"rule":"degree", "scalecoeff":0.00001},
            nodeCoords=node_coords,
            edgeColorDict=edge_colors,
            defaultEdgeAlpha=0.25,
            layerColorDict=layer_colors,
            defaultLayerAlpha=0.075,
            layerLabelRule={},  # Clear any label rules
            defaultLayerLabel=None,  # Set default label to None
            azim=45,
            elev=25)

    print(type(fig))

    return fig

# %%

def multiplex_diff_viz(M, weighted_G, ax=None, node_colors=None, node_sizes=None):
    # Load the pickle file
    with open('diff_results/Pathway_False_target_BB_GDDs_ks272.pkl', 'rb') as f:
        results = pkl.load(f)
    
    time_resolved_kernels = results['BIRC2'][0.05]['vis_kernels'][:12]


    # Assume 'weighted_G_cms_ALL' is your graph
    node_order = list(weighted_G.nodes())
    # get indices of nodes that end with '.t'
    t_indices = [i for i, node in enumerate(node_order) if node.endswith('.t')]
    #consistent node positions
    prot_nodes = [node for node in weighted_G.nodes() if node.endswith('.p')]
    prot_node_positions = nx.spring_layout(weighted_G.subgraph(prot_nodes))  # Adjust as necessary

    # Set up a 3x3 subplot grid with 3D projection
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20), subplot_kw={'projection': '3d'})
    axs = axes.flatten()  # Flatten the axes array for easy iteration
    fig.suptitle('Network Diffusion over 25 Time Points')

    # Define the suffixes for each layer
    suffixes = {'PROTEIN': '.p', 'RNA': '.t'}

    # Create an index mapping from the suffixed node name to its index
    node_indices = {node: i for i, node in enumerate(node_order)}
    node_colors = {}  
    node_sizes = {} 
    node_coords = {}

    for node, pos in prot_node_positions.items():
        stripped_node = node.rstrip('.t')  # Remove the suffix to match identifiers in M
        node_coords[stripped_node] = pos

    for nl in M.iter_node_layers():  # Iterating over all node-layer combinations
        node, layer = nl  # Split the node-layer tuple
        neighbors = list(M._iter_neighbors_out(nl, dims=None))  # Get all neighbors for the node-layer tuple
        degree = len(neighbors)  # The degree is the number of neighbors
        
        # Assign to node sizes
        node_sizes[nl] = 0.0001 * degree**2  # 0.0015 * degree # Adjust the scaling factor as needed

    print(node_sizes)


    j = 271
    global_max = max(kernel.max() for kernel in time_resolved_kernels)
    norm = Normalize(vmin=0, vmax=1)
    # print(global_max)
    # Ensure you have a list of the 25 time-resolved kernels named 'time_resolved_kernels'
    for idx, (ax, kernel) in enumerate(zip(axs, time_resolved_kernels)):
        # if idx % 5 == 0:
        # Create the unit vector e_j with 1 at the jth index and 0 elsewhere
        e_j = np.zeros(len(weighted_G.nodes()))
        e_j[t_indices] = 1

        # e_j = np.ones(len(weighted_G_cms_ALL.nodes()))
        
        # Multiply the kernel with e_j to simulate diffusion from node j
        diffusion_state = kernel @ e_j
        # order the diffusion state in descending order
        diffusion_state = sorted(diffusion_state, reverse=True)

        # Now, update node colors and sizes based on diffusion_state
        for layer in M.iter_layers():  # Iterates through all nodes in M
            # Determine the layer of the node for color and size settings
            for node in M.iter_nodes(layer=layer):
                # Append the appropriate suffix to the node name to match the format in node_order
                suffixed_node = node + suffixes[layer]
                # Use the suffixed node name to get the corresponding index from the node_order
                index = node_indices[suffixed_node]

                # Map the diffusion state to a color and update node_colors and node_sizes
                color = plt.cm.viridis(diffusion_state[index])  # Mapping color based on diffusion state
                node_colors[(node, layer)] = color
                # node_sizes[(node, layer)] = 0.03  # Or some logic to vary size with diffusion state

        # Now use your updated visualization function with the new colors and sizes
        diff_fig = multiplex_net_viz(M, ax, diff_colors=True, node_colors=node_colors, node_sizes=node_sizes, node_coords=node_coords)

        ax.set_title(f"Time Step {idx*20}")

        # ax.imshow(diff_fig)

        # # Draw the graph with node colors based on the diffusion state
        # nx.draw(weighted_G_cms_ALL, pos, ax=ax, node_size=50,
        #         node_color=norm(diffusion_state), cmap=plt.cm.viridis,
        #         edge_color=(0, 0, 0, 0.5), width=1.0)  # Use a simple color for edges for clarity

        # # # print maximum value of kernel at this time step
        # # print(f'max kernel value at time step {idx}: {kernel.max()}')

        # # Set a title for each subplot indicating the time step
        # ax.set_title(f"Time Step {idx*20}")

    # Adjust layout to prevent overlap
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.95)  # Adjust the top space to fit the main title
    # plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), ax=axs.ravel().tolist(), orientation='vertical')

    # savethefigure
    plt.savefig('diffusion_figure.png', dpi=300)

    # Display the figure
    plt.show()

if args.visualize:
    multiplex_diff_viz(pymnet_ALL, weighted_G_cms_ALL)


# %% 
   
    # plt.savefig('network_figure.png', dpi=300)

# multiplex_diff_viz(pymnet_ALL)
# # %% OLD CODE FOR GDD PLOTTING AND MAXIMUM GDD VALUE SORTING
# # t_values = np.linspace(0.01, 10, 500)
# # red_range = red_args.split(',')
# # # red_range = [float(i) for i in red_range]
# # red_range = [float(red_range[0]), float(red_range[1]), int(float(red_range[2]))]

# kernel_size = 272
# Pathway = True

# t_values = np.linspace(0.01, 10, 500)
# red_range = args.red_range.split(',')
# red_range = np.linspace(float(red_range[0]), float(red_range[1]), int(float(red_range[2])))

# filename = f'diff_results/Pathway_{Pathway}_GDDs_{kernel_size}.pkl'


# if not "SLURM_JOB_ID" in os.environ:
#     with open(filename, 'rb') as f:
#         GDDs_and_Kernels = pkl.load(f)

#     first_key_outer = next(iter(GDDs_and_Kernels))
#     first_key_inner = next(iter(GDDs_and_Kernels[first_key_outer]))

#     print(GDDs_and_Kernels[first_key_outer][first_key_inner].keys())

#     orig_max_gdd = GDDs_and_Kernels[first_key_outer][first_key_inner]['max_gdd_orig']
    
#     print(f'GDDs_and_Kernels: {GDDs_and_Kernels.keys()}')
#     print(f'Reduction factors: {GDDs_and_Kernels[list(GDDs_and_Kernels.keys())[0]].keys()}')
#     # Choose a reduction factor from the list of reductions
#     selected_reduction = red_range[1]

#     # Calculate max GDDs for each target (node or pathway) and sort them
#     max_gdds_trans = {target: np.max(GDDs_and_Kernels[target][selected_reduction]['gdd_values_trans']) for target in GDDs_and_Kernels}
#     max_gdds_disrupt = {target: np.max(GDDs_and_Kernels[target][selected_reduction]['gdd_values_disrupt']) for target in GDDs_and_Kernels}

#     sorted_max_gdds_trans = sorted(max_gdds_trans.items(), key=lambda item: item[1])
#     sorted_max_gdds_disrupt = sorted(max_gdds_disrupt.items(), key=lambda item: item[1], reverse=True)

#     # Calculate the percentage of the original max GDD for each sorted max GDD
#     max_gdds_trans_percent = {target: (value / orig_max_gdd) * 100 for target, value in sorted_max_gdds_trans}
#     max_gdds_disrupt_percent = {target: 'N/A' for target, value in sorted_max_gdds_disrupt}

#     half_point = int(len(sorted_max_gdds_disrupt) / 2)
#     targets_to_show = 3

#     # Select top 3 and bottom 3 targets for each case
#     top_3_trans = [target for target, _ in sorted_max_gdds_trans[-targets_to_show:]]
#     bottom_3_trans = [target for target, _ in sorted_max_gdds_trans[:targets_to_show]]
#     top_3_disrupt = [target for target, _ in sorted_max_gdds_disrupt[-targets_to_show:]]
#     bottom_3_disrupt = [target for target, _ in sorted_max_gdds_disrupt[:targets_to_show]]

#     fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=300)  # Creating 4 plots
#     (ax1, ax2), (ax3, ax4) = axes

#     # Plot GDD values for top 3 and bottom 3 targets (Trans)
#     for target in top_3_trans + bottom_3_trans:
#         gdd_values_trans = GDDs_and_Kernels[target][selected_reduction]['gdd_values_trans']
#         ax1.plot(t_values, gdd_values_trans, label=f'target {target}')

#     # Plot GDD values for top 3 and bottom 3 targets (Disrupt)
#     for target in top_3_disrupt + bottom_3_disrupt:
#         gdd_values_disrupt = GDDs_and_Kernels[target][selected_reduction]['gdd_values_disrupt']
#         ax3.plot(t_values, gdd_values_disrupt, label=f'target {target}')

#     ax1.set_title(f'GDD Over Time (Trans)\nTop 3 and Bottom 3 targets')
#     ax1.set_xlabel('Time')
#     ax1.set_ylabel('GDD Value (Trans)')
#     ax1.legend()
#     ax1.grid(True)

#     ax3.set_title(f'GDD Over Time (Disrupt)\nTop 3 and Bottom 3 targets')
#     ax3.set_xlabel('Time')
#     ax3.set_ylabel('GDD Value (Disrupt)')
#     ax3.legend()
#     ax3.grid(True)

#     plt.show()

# def write_to_csv(data, percent_data, filename):
#     """
#     Writes the data to a CSV file with three columns.
#     :param data: List of tuples, where each tuple contains two elements (key, value)
#     :param percent_data: Dictionary with the percentage values
#     :param filename: Name of the CSV file to be written
#     """
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         for key, value in data:
#             writer.writerow([key, value, percent_data[key]])

# # Write to CSV files
# write_to_csv(sorted_max_gdds_trans, max_gdds_trans_percent, f'diff_results/max_gdds_trans_Pathway_{Pathway}_{kernel_size}.csv')
# write_to_csv(sorted_max_gdds_disrupt, max_gdds_disrupt_percent, f'diff_results/max_gdds_disrupt_Pathway_{Pathway}_{kernel_size}.csv')






# %%
# cms = 'cmsALL'
# kernel_size = 20

# if not "SLURM_JOB_ID" in os.environ:
#     with open(f'diff_results/GDDs_and_Kernels_268.pkl', 'rb') as f:
#         GDDs_and_Kernels = pkl.load(f)

#     print(f'GDDs_and_Kernels: {GDDs_and_Kernels.keys()}')
#     print(f'Reduction factors: {GDDs_and_Kernels[list(GDDs_and_Kernels.keys())[0]].keys()}')

#     # Choose the node and t_values for plotting
#     selected_node = np.random.choice(list(GDDs_and_Kernels.keys()))

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

#     # Left plot: GDD values over time for various reductions (single node)
#     for reduction in GDDs_and_Kernels[selected_node].keys():
#         gdd_values = GDDs_and_Kernels[selected_node][reduction]['gdd_values']
#         ax1.plot(t_values, gdd_values, label=f'Reduction {reduction}')

#     ax1.set_title(f'GDD Over Time for Various Reductions\nNode: {selected_node}')
#     ax1.set_xlabel('Time')
#     ax1.set_ylabel('GDD Value')
#     ax1.legend()
#     ax1.grid(True)

#     # Choose a reduction factor from the list of reductions
#     selected_reduction = red_range[1]

#     max_gdds = {}
#     # Right plot: GDD values over time for a single reduction (all nodes)
#     for node_base in GDDs_and_Kernels.keys():
#         gdd_values = GDDs_and_Kernels[node_base][selected_reduction]['gdd_values']
#         ax2.plot(t_values, gdd_values, label=f'Node {node_base}', alpha=0.5)
#         max_gdds[node_base] = np.max(gdd_values)

#     ax2.set_title(f'GDD Over Time for Single Reduction\nReduction: {selected_reduction}')
#     ax2.set_xlabel('Time')
#     # ax2.set_ylabel('GDD Value')  # Y-label is shared with the left plot
#     # ax2.legend()
#     ax2.set_xlim([0, 2])
#     ax2.grid(True)

#     plt.show()
#     # print(max_gdds)
#     # max_GDD_1 = max_gdds['1']
#     # max_GDD_2 = max_gdds['2']
#     # print(max_GDD_1 - max_GDD_2)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=300)

#     # PLOT 2 (WEAKER KNOCKDOWN)
#     # make another plot as ax2 but for reduction factor = 0.8
#     selected_reduction = red_range[-1]

#     max_gdds = {}
#     # Right plot: GDD values over time for a single reduction (all nodes)
#     for node_base in GDDs_and_Kernels.keys():
#         gdd_values = GDDs_and_Kernels[node_base][selected_reduction]['gdd_values']
#         ax2.plot(t_values, gdd_values, label=f'Node {node_base}', alpha=0.5)
#         max_gdds[node_base] = np.max(gdd_values)


#     ax2.set_title(f'GDD Over Time for Single Reduction\nReduction: {selected_reduction}')
#     ax2.set_xlabel('Time')
#     # ax2.set_ylabel('GDD Value')  # Y-label is shared with the left plot
#     # ax2.legend()
#     ax2.set_xlim([0, 2])
#     ax2.grid(True)

#     plt.show()
#     # print(max_gdds)
#     # max_GDD_1 = max_gdds['1']
#     # max_GDD_2 = max_gdds['2']
#     # print(max_GDD_1 - max_GDD_2)

# selected_reduction = red_range[-1]
# # order nodes by max GDD
# max_gdds = {}
# for node_base in GDDs_and_Kernels.keys():
#     max_gdds[node_base] = np.max(GDDs_and_Kernels[node_base][selected_reduction]['gdd_values'])

# sorted_max_gdds = {k: v for k, v in sorted(max_gdds.items(), key=lambda item: item[1])}

# # get the nodes with the highest GDD
# highest_gdd_nodes = list(sorted_max_gdds.keys())[-5:]
# highest_gdd_nodes

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
# knockdown_graph, knockdown_laplacian = knockdown_node(weighted_graph_use, node_to_isolate, reduced_weight=fixed_reduction)

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
