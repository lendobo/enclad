from estimate_lambdas import estimate_lambda_np, estimate_lambda_wp, find_all_knee_points
from piglasso import QJSweeper
from evaluation_of_graph import optimize_graph, evaluate_reconstruction

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def run(data, 
        prior_matrix, 
        p, 
        n, 
        Q,
        lambda_range, 
        lowerbound, 
        upperbound, 
        granularity, 
        edge_counts_all, 
        prior_bool=False,
        adj_matrix=None, 
        run_type='SYNTHETIC',
        plot=False):

    # KNEE POINTS
    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)

    l_lo = left_knee_point_index     # Set it at knee-point index -1 
    if run_type == 'SYNTHETIC':
        l_hi = right_knee_point_index + 10 # set at knee-point index
    else:
        l_hi = right_knee_point_index + 25
        print(f'right_knee_point_index: {right_knee_point_index}')
        print(f'selected l_hi: {lambda_range[l_hi]}')
    
    print(f'{run_type} RESULTS\n-------------------\n')

    select_lambda_range = lambda_range[l_lo:l_hi]
    print(f'Selected lambda range: {select_lambda_range[0]} - {select_lambda_range[-1]}')
    select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]


    # LAMBDAS
    lambda_np, theta_mat = estimate_lambda_np(select_edge_counts_all, Q, select_lambda_range)
    print('lambda_np: ', lambda_np)
    if prior_bool == True:
        lambda_wp, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, prior_matrix)
        print('lambda_wp: ', lambda_wp, '\n')
    else:
        lambda_wp = 0
        print('lambda_wp: ', lambda_wp, '\n')


    # GRAPH OPTIMIZATION WITH FOUND LAMBDAS
    precision_matrix, edge_counts, density = optimize_graph(data, prior_matrix, lambda_np, lambda_wp)

    print('Number of edges of inferred network: ', edge_counts)
    print('Density: ', density)

    if run_type == 'SYNTHETIC':
        # RECONSTRUCTION
        evaluation_metrics = evaluate_reconstruction(adj_matrix, precision_matrix)
        print('Evaluation metrics: ', evaluation_metrics)

    if plot == True:
        scalar_edges = np.sum(edge_counts_all, axis=(0, 1))
        scalar_select_edges = np.sum(select_edge_counts_all, axis=(0, 1))

        # create a 1 x 2 multiplot. on the left, plot both scalar aedes and scalar_select edges. On the right, just scalar_select_edges
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(lambda_range, scalar_edges, color='grey', alpha = 0.5)
        plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
        plt.title(f'Number of edges vs lambda for {run_type} data')
        plt.xlabel('Lambda')
        plt.ylabel('Number of edges')
        plt.grid()
        ax = plt.gca()
        ax.grid(alpha=0.2)

        plt.subplot(1, 2, 2)
        plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
        plt.title(f'Number of edges vs lambda for {run_type} data')
        plt.xlabel('Lambda')
        plt.ylabel('Number of edges')
        plt.grid()
        ax = plt.gca()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    return precision_matrix, edge_counts, density
    



rank=1
size=1

# ################################################# SYNTHETIC PART #################################################
# # Parameters
# p = 100             # number of variables (nodes)
# n = 800             # number of samples
# b = int(0.75 * n)   # size of sub-samples
# Q = 1200             # number of sub-samples

# lowerbound = 0.01
# upperbound = 0.4
# granularity = 60
# lambda_range = np.linspace(lowerbound, upperbound, granularity)

# # Load edge counts
# filename_edges = f'net_results/synthetic_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_{granularity}.pkl'
# with open(filename_edges, 'rb') as f:
#     synth_edge_counts_all = pickle.load(f)
# # divide each value in edge_counts_all by 2*Q
# synth_edge_counts_all = synth_edge_counts_all / (2 * Q)

# # # Generate synthetic data and prior matrix
# synth_data, synth_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n)


# ## REsults for synthetic data
# run(synth_data, synth_prior_matrix, p, n, Q, lowerbound, upperbound, granularity, synth_edge_counts_all, adj_matrix=synth_adj_matrix, run_type='SYNTHETIC', plot=True)


################################################## OMICS DATA PART #################################################
# Parameters
p = 50
Q = 800             # number of sub-samples

lowerbound = 0.01
upperbound = 0.55
granularity = 60
lambda_range = np.linspace(lowerbound, upperbound, granularity)


o_t = 'p'
if o_t == 'p':
    prior_bool = True
    omics_type = 'proteomics'
elif o_t == 't':
    prior_bool = False
    omics_type = 'transcriptomics'


# Load omics edge counts
filename_edges = f'Networks/net_results/local_{omics_type}_edge_counts_all_pnQ{p}_500_800_0.01_0.7_60.pkl'
with open(filename_edges, 'rb') as f:
    omics_edge_counts_all = pickle.load(f)

# divide each value in edge_counts_all by 2*Q
omics_edge_counts_all = omics_edge_counts_all / (2 * Q)


# Load Omics Data
cms_data = pd.read_csv(f'Diffusion/data/TCGACRC_{omics_type}_SUBGRAPH_SELECT_50.csv', index_col=0)
cms_array = cms_data.values

# LOad Omics Prior Matrix
if prior_bool == True:
    cms_omics_prior = pd.read_csv('Diffusion/data/SUBGRAPH_ADJACENCY_PROT_50.csv', index_col=0)
else:
    cms_omics_prior = pd.read_csv('Diffusion/data/SUBGRAPH_ADJACENCY_PROT_50.csv', index_col=0)
    #only keep columns / rows that are in the omics data
    cms_omics_prior = cms_omics_prior[cms_data.columns]
    cms_omics_prior = cms_omics_prior.reindex(index=cms_data.columns)
    cms_omics_prior = cms_omics_prior * 0

cms_omics_prior_matrix = cms_omics_prior.values
# Check if there are any non-zero values in the prior matrix
print(f'edges in prior: {np.sum(cms_omics_prior_matrix != 0) / 2}')

p = cms_array.shape[1]
n = cms_array.shape[0]
b = int(0.75 * n)

# scale and center 
cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(axis=0)

# RUN ANALYSIS
precision_mat, edge_counts, density = run(cms_array, cms_omics_prior_matrix, p, n, Q, lambda_range, 
            lowerbound, upperbound, granularity, omics_edge_counts_all, prior_bool, run_type='OMICS', plot=True)


# get adjacency from precision matrix
adj_matrix = (np.abs(precision_mat) > 1e-5).astype(int)
# assign columns and indices of prior matrix to adj_matrix
adj_matrix = pd.DataFrame(adj_matrix, index=cms_data.columns, columns=cms_data.columns)
print(adj_matrix.head())

# save adjacency matrix
adj_matrix.to_csv(f'Networks/net_results/{omics_type}_adj_matrix_pnQ{p}_500_800_0.01_0.7_60.csv')