# %%
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (MONIKA)
project_dir = os.path.dirname(script_dir)

# Add the project directory to the Python path
sys.path.append(project_dir)

# Change the working directory to the project directory
os.chdir(project_dir)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from diffupy.diffuse import run_diffusion_algorithm
from diffupy.matrix import Matrix
from diffupy.diffuse_raw import diffuse_raw
from diffupy.kernels import regularised_laplacian_kernel, diffusion_kernel
import random

# %%
# switch matplotlib background to dark mode
plt.style.use('dark_background')


# %%

def STRING_adjacency_matrix(nodes_df, edges_df):
    """
    Generate an adjacency matrix from the edgelist and nodelist obtained from STRING database. 
    """
    # Mapping Ensembl IDs to 'query term' names
    id_to_query_term = pd.Series(nodes_df['query term'].values, index=nodes_df['name']).to_dict()

    # Create a unique list of 'query terms'
    unique_query_terms = nodes_df['query term'].unique()

    # Initialize an empty adjacency matrix with unique query term labels
    adjacency_matrix = pd.DataFrame(0, index=unique_query_terms, columns=unique_query_terms)

    # Process each edge in the edges file
    for _, row in edges_df.iterrows():
        # Extract Ensembl IDs from the edge and map them to 'query term' names
        gene1_id, gene2_id = row['name'].split(' (pp) ')
        gene1_query_term = id_to_query_term.get(gene1_id)
        gene2_query_term = id_to_query_term.get(gene2_id)

        # Check if both gene names (query terms) are in the list of unique query terms
        if gene1_query_term in unique_query_terms and gene2_query_term in unique_query_terms:
            # Set the undirected edge in the adjacency matrix
            adjacency_matrix.loc[gene1_query_term, gene2_query_term] = 1
            adjacency_matrix.loc[gene2_query_term, gene1_query_term] = 1


    return adjacency_matrix


# Loading Edges and Nodes with 85% or above confidence according to STRING
STRING_edges_df = pd.read_csv('data/prior_data/RPPA_prior_EDGES90perc.csv')
STRING_nodes_df = pd.read_csv('data/prior_data/RPPA_prior_NODES90perc.csv')


# # Construct the adjacency matrix from STRING
PPI_interactions = STRING_adjacency_matrix(STRING_nodes_df, STRING_edges_df)

PPI_interactions.head()
# write to csv
PPI_interactions.to_csv('data/prior_data/RPPA_prior_adj90perc.csv', index=True)
