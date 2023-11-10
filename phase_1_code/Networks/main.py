from estimate_lambdas import estimate_lambda_np, estimate_lambda_wp, find_all_knee_points
from old_piGGM import QJSweeper
from evaluation_of_graph import optimize_graph, evaluate_reconstruction

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

p = 100             # number of variables (nodes)
n = 800             # number of samples
b = int(0.75 * n)   # size of sub-samples
Q = 1200             # number of sub-samples

lowerbound = 0.01
upperbound = 0.4
granularity = 60
lambda_range = np.linspace(lowerbound, upperbound, granularity)


rank=1
size=1

################################################# SYNTHETIC PART #################################################
# Load edge counts
filename_edges = f'net_results/synthetic_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_{granularity}.pkl'
with open(filename_edges, 'rb') as f:
    edge_counts_all = pickle.load(f)

# divide each value in edge_counts_all by 2*Q
edge_counts_all = edge_counts_all / (2 * Q)

# # Generate synthetic data and prior matrix
synth_data, prior_matrix, adj_matrix = QJSweeper.generate_synth_data(p, n)

print('SYNTHETIC RESULTS\n-------------------\n')
# KNEE POINTS
left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)

l_lo = left_knee_point_index     # Set it at knee-point index -1 
l_hi = right_knee_point_index + 10   # set at knee-point index


select_lambda_range = lambda_range[l_lo:l_hi]
print(f'Selected lambda range: {select_lambda_range[0]} - {select_lambda_range[-1]}')
select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]

# LAMBDAS
lambda_np, theta_mat = estimate_lambda_np(select_edge_counts_all, Q, select_lambda_range)
lambda_wp, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, prior_matrix)
print('lambda_np: ', lambda_np)
print('lambda_wp: ', lambda_wp)

# lambda_np = 1
# lambda_wp = 0.04

# GRAPH OPTIMIZATION WITH FOUND LAMBDAS
precision_matrix, edge_counts, density = optimize_graph(synth_data, prior_matrix, lambda_np, lambda_wp)

print('Number of edges: ', edge_counts)
print('Density: ', density)



# RECONSTRUCTION
evaluation_metrics = evaluate_reconstruction(adj_matrix, precision_matrix)
print('Evaluation metrics: ', evaluation_metrics)


# ################################################# OMICS DATA PART #################################################
print('OMICS RESULTS\n-------------------\n')
# ESTIMATING LAMBDA FOR OMICS DATA
# Load omics data and prior matrix
filename_edges = 'net_results/omics_edge_counts_all_pnQ100_800_300_0.01_0.6_60.pkl'
with open(filename_edges, 'rb') as f:
    edge_counts_all = pickle.load(f)

lambda_np, theta_mat = estimate_lambda_np(edge_counts_all, Q, lambda_range)
print('lambda_np: ', lambda_np)

# lambda_wp_t, tau_tr, mus = estimate_lambda_wp(edge_counts_all, Q, lambda_range, prior_matrix_t)
# print('lambda_wp_t: ', lambda_wp_t)


# GRAPH OPTIMIZATION WITH FOUND LAMBDAS
cms2_data = pd.read_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms2_top_DEA.csv', index_col=0)

cms2_data = cms2_data.iloc[:, :p]
cms2_array = cms2_data.values

n = cms2_array.shape[0]
b = int(0.75 * n)

# scale and center 
cms2_array = (cms2_array - cms2_array.mean(axis=0)) / cms2_array.std(axis=0)
prior_matrix = np.zeros((p, p))

# Optimize the graph
edge_counts, precision_matrix, success = optimize_graph(cms2_array, prior_matrix, lambda_np)

# get total number of edge counts by summing across both axes
num_edges = np.sum(edge_counts) / 2
print(f'Number of edges: {num_edges}')

complete_g = p * (p - 1) / 2

# get the percentage of edges that are present
edge_percentage = num_edges / complete_g
print(f'Percentage of edges: {edge_percentage}')

