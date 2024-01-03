import random
import numpy as np
import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
import pymnet as pn


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

    if plot and False:
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

    if plot:
        # plot the netowrk
        pos = nx.spring_layout(G_multiplex)
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_nodes(G_multiplex, pos, node_size=10)
        nx.draw_networkx_edges(G_multiplex, pos, alpha=0.5)
        plt.show()

    M = 0
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

from pymnet import *
