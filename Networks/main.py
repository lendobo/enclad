from estimate_lambdas import estimate_lambda_np, estimate_lambda_wp, find_all_knee_points
from piglasso import QJSweeper
from evaluation_of_graph import optimize_graph, evaluate_reconstruction

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stats
from collections import Counter

def analysis(data, 
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
        kneepoint_adder=0,
        plot=False):

    # KNEE POINTS
    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)

    l_lo = left_knee_point_index     # Set it at knee-point index -1 
    if run_type == 'SYNTHETIC':
        l_hi = right_knee_point_index # set at knee-point index
    else:
        l_hi = right_knee_point_index + kneepoint_adder                                                                                 
        print(f'\nADDER (right): +{kneepoint_adder}')
        print(f'right_knee_point_index: {right_knee_point_index}')
        # print(f'selected l_hi: {lambda_range[l_hi]}')
    
    print(f'complete lambda range: {lowerbound, upperbound}')                                                              # HERE
    select_lambda_range = lambda_range[l_lo:l_hi]
    print(f'Selected lambda range: {select_lambda_range[0]} - {select_lambda_range[-1]} \n')
    select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]


    # LAMBDAS
    lambda_np, theta_mat = estimate_lambda_np(select_edge_counts_all, Q, select_lambda_range)
    man= False
    if man:
        lambda_np =  0.136
        print(f'manual Lambda_np: {man}')
    print('lambda_np: ', lambda_np)
    if prior_bool == True:
        lambda_wp, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, prior_matrix)
        # lambda_wp = 0.076
        print('lambda_wp: ', lambda_wp, '\n')
    else:
        lambda_wp = 0
        print('lambda_wp: ', lambda_wp, '\n')


    # GRAPH OPTIMIZATION WITH FOUND LAMBDAS
    precision_matrix, edge_counts, density = optimize_graph(data, prior_matrix, lambda_np, lambda_wp)

    print('Number of edges of inferred network (lower triangular): ', edge_counts)
    print('Density: ', density)

    if run_type == 'SYNTHETIC':
        # RECONSTRUCTION
        evaluation_metrics = evaluate_reconstruction(adj_matrix, precision_matrix)
        print('Evaluation metrics: ', evaluation_metrics)
        print('\n\n\n')

    if plot == True:
        scalar_edges = np.sum(edge_counts_all, axis=(0, 1))
        scalar_select_edges = np.sum(select_edge_counts_all, axis=(0, 1))

        # create a 1 x 2 multiplot. on the left, plot both scalar aedes and scalar_select edges. On the right, just scalar_select_edges
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(lambda_range, scalar_edges, color='grey', alpha = 0.5)
        plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
        plt.title(f'#edges vs lambda for {run_type} data,p={p},n={n}')
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
    
    return precision_matrix, edge_counts, density, lambda_np, lambda_wp
    




rank=1
size=1
# ################################################# SYNTHETIC PART #################################################

################################################## VARYING N #################################
if False:
    # Parameters
    p = 150             # number of variables (nodes)
    n = 500             # number of samples
    b = int(0.8 * n)   # size of sub-samples
    Q = 1000             # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.4
    granularity = 80
    lambda_range = np.linspace(lowerbound, upperbound, granularity)

    n = 69
    p_values = [50, 100, 200, 400]

    for p in p_values:
        print(f'NETWORK SIZE for samples: {n} and variables: {p}')
        filename_edges = f'Networks/net_results/synthetic_cmsALL_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{granularity}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{density}.pkl'
        with open(filename_edges, 'rb') as f:
            synth_edge_counts_all = pickle.load(f)
        # divide each value in edge_counts_all by 2*Q
        synth_edge_counts_all = synth_edge_counts_all / (2 * Q)

        # # Generate synthetic data and prior matrix
        synth_data, synth_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n, skew=skew, fp_fn_chance=fp_fn, density=density)
        ## REsults for synthetic data
        # print(f'SYNTHETIC NET SIZE: {p} RESULTS\n-------------------\n')

        analysis(synth_data, synth_prior_matrix, p, n, Q, lambda_range, lowerbound, upperbound, granularity, 
        synth_edge_counts_all, prior_bool=True, adj_matrix=synth_adj_matrix, run_type='SYNTHETIC', plot=False)
    
if True:
    ################################################# VARYING SAMPLE SIZE for fixed network #################################################
    p = 137
    n_values = [500] #[100, 200, 400, 600, 800]
    b_perc = 0.6
    b = [int(b_perc * n) for n in n_values]   # size of sub-samples
    Q = 1200             # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.5
    granularity = 100
    lambda_range = np.linspace(lowerbound, upperbound, granularity)

    fp_fn = 0.0
    skew = 0
    density = 0.03

    for n,b in zip(n_values, b):
        print(f'NETWORK SIZE for samples: {n} and variables: {p} and b: {b}')
        filename_edges = f'Networks/net_results/synthetic_cmsALL_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{granularity}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{density}.pkl'
        with open(filename_edges, 'rb') as f:
            synth_edge_counts_all = pickle.load(f)
        # divide each value in edge_counts_all by 2*Q
        synth_edge_counts_all = synth_edge_counts_all / (2 * Q)  # EDGE_DIVIDER

        # # Generate synthetic data and prior matrix
        synth_data, synth_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n, skew=skew, fp_fn_chance=fp_fn, density=density)
        # synth_prior_matrix = synth_prior_matrix * 0

        analysis(synth_data, synth_prior_matrix, p, n, Q, lambda_range, lowerbound, upperbound, granularity, 
        synth_edge_counts_all, prior_bool=True, adj_matrix=synth_adj_matrix, run_type='SYNTHETIC', plot=False)


        



################################################## OMICS DATA PART #################################################
if False:
    # Parameters
    p = 137
    Q = 800             # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.8
    granman = 100

    o_t = 'p'
    cms = 'cmsALL' #cmsALL

    if o_t == 'p':
        prior_bool = True
        omics_type = 'proteomics'
    elif o_t == 't':
        prior_bool = False
        omics_type = 'transcriptomics'


    # Load omics edge counts
    file_ = f'Networks/net_results/{omics_type}_{cms}_edge_counts_all_pnQ150_200_1000_{lowerbound}_{upperbound}_{granman}_10.pkl'

    with open(file_, 'rb') as f:
        omics_edge_counts_all = pickle.load(f)

    # divide each value in edge_counts_all by 2*Q
    omics_edge_counts_all = omics_edge_counts_all / (2)

    # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
    granularity = omics_edge_counts_all.shape[2]
    lambda_range = np.linspace(lowerbound, upperbound, granularity)


    # Load Omics Data
    cms_data = pd.read_csv(f'Diffusion/data/{omics_type}_for_pig_{cms}.csv', index_col=0)

    cms_array = cms_data.values

    print(f'Number of samples: {cms_array.shape[0]}')


    # remove rows if they contain values outside of 3 standard deviations
    cms_array = cms_array[(np.abs(stats.zscore(cms_array)) < 5).all(axis=1)]

    print(f'Number of samples: {cms_array.shape[0]}')



    # LOad Omics Prior Matrix
    if prior_bool == True:
        cms_omics_prior = pd.read_csv('Diffusion/data/RPPA_prior_adj.csv', index_col=0)
    else:
        cms_omics_prior = pd.read_csv('Diffusion/data/RPPA_prior_adj.csv', index_col=0)
        #only keep columns / rows that are in the omics data
        cms_omics_prior = cms_omics_prior[cms_data.columns]
        cms_omics_prior = cms_omics_prior.reindex(index=cms_data.columns)
        cms_omics_prior = cms_omics_prior * 0

    cms_omics_prior_matrix = cms_omics_prior.values
    # # Check if there are any non-zero values in the prior matrix
    # print(f'edges in prior: {np.sum(cms_omics_prior_matrix != 0) / 2}')

    p = cms_array.shape[1]
    n = cms_array.shape[0]
    b = int(0.8 * n)

    # scale and center 
    cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(axis=0)

    print(f'{str.upper(omics_type)}, {cms} RESULTS\n-------------------\n')


    print(f'Number of samples: {n}')
    print(f'Number of variables: {p}\n')

    print(f'Granularity of full lambda range: {granularity}')

    # # RUN ANALYSIS
    kpa = 0                                                                                                        # HERE
    precision_mat, edge_counts, density, lambda_np, lambda_wp = analysis(cms_array, cms_omics_prior_matrix, p, n, Q, lambda_range, 
                lowerbound, upperbound, granularity, omics_edge_counts_all, prior_bool, run_type='OMICS', kneepoint_adder=kpa, plot=False)


    # get adjacency from precision matrix
    adj_matrix = (np.abs(precision_mat) > 1e-5).astype(int)
    # assign columns and indices of prior matrix to adj_matrix
    adj_matrix = pd.DataFrame(adj_matrix, index=cms_data.columns, columns=cms_data.columns)

    # save adjacency matrix
    adj_matrix.to_csv(f'Networks/net_results/inferred_adjacencies/{omics_type}_{cms}_adj_matrix_p{p}_kpa{kpa}_np{lambda_np}.csv')

    # # draw the network
    # G = nx.from_pandas_adjacency(cms_omics_prior)
    # nx.draw(G, with_labels=True)
    # plt.title(f'Network for {omics_type} data')
    # plt.show()

    # #plot the degree distribution
    G = nx.from_pandas_adjacency(adj_matrix)
    degrees = [G.degree(n) for n in G.nodes()]
    # plt.hist(degrees, bins=20)
    # plt.title(f'Degree distribution for {omics_type} data')
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.show()

    highest_degrees_indices = list(np.argsort(degrees)[-20:])
    nodes_with_highest_degrees = [list(G.nodes())[i] for i in highest_degrees_indices]

    print(f'Highest degrees: {np.sort(degrees)[-20:]}')
    print(f'Nodes with highest degrees: {nodes_with_highest_degrees}')

    # Print the degree of 'TP53'
    print(f'Degree of TP53: {G.degree("TP53")}')



    # # LOG - LOG SCALE   
    # # Count the frequency of each degree
    # degree_counts = Counter(degrees)
    # degrees, counts = zip(*degree_counts.items())
    # # Scatter plot
    # plt.scatter(degrees, counts)
    # # Set both axes to logarithmic scale
    # plt.xscale('log')
    # plt.yscale('log')
    # # Set the labels and title
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.title(f'Log-Log Scatter Plot of Degree Distribution for {omics_type} data')
    # # Show the plot
    # plt.show()






# # TESTING FOR NORMALITY
# # perform shapiro-wilk test on each column
# for i in range(cms_array.shape[1]):
#     result = stats.shapiro(cms_array[:, i])
#     print(result)

# # get first 20 columns
# cms_array = cms_array[:, 36]
# # make QQ plot for this column
# plt.figure(figsize=(12, 5))
# stats.probplot(cms_array, dist="norm", plot=plt)
# plt.title(f'Column {i+1}')
# plt.tight_layout()
# plt.show()

# # print lowest value in this column
# print(np.min(cms_array))
# # print highest value in this column
# print(np.max(cms_array)) 

# # make QQ plots in a multiplot for the first 20 columns
# plt.figure(figsize=(12, 5))
# for i in range(cms_array.shape[1]):
#     plt.subplot(2, 5, i+1)
#     stats.probplot(cms_array[:, i], dist="norm", plot=plt)
#     plt.title(f'Column {i+1}')
# plt.tight_layout()
# plt.show()