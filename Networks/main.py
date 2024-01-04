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

from collections import defaultdict
import os

from tqdm import tqdm

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
        plot=False,
        verbose=False):

    # KNEE POINTS
    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)

    l_lo = left_knee_point_index     # Set it at knee-point index -1 
    if run_type == 'SYNTHETIC':
        l_hi = right_knee_point_index # set at knee-point index
    else:
        l_hi = right_knee_point_index + kneepoint_adder  
        if verbose:
            print(f'\nADDER (right): + {kneepoint_adder}')
            print(f'right_knee_point_index: {right_knee_point_index}')
        # print(f'selected l_hi: {lambda_range[l_hi]}')
    

    select_lambda_range = lambda_range[l_lo:l_hi]
    select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]

    if verbose:
        print(f'complete lambda range: {lowerbound, upperbound}')                                                              # HERE
        print(f'Selected lambda range: {select_lambda_range[0]} - {select_lambda_range[-1]} \n')

    # LAMBDAS
    lambda_np, theta_mat = estimate_lambda_np(select_edge_counts_all, Q, select_lambda_range)
    man = False
    if run_type == 'OMICS':
        man= True
        if man:
            lambda_np =  1
            if verbose:
                # print('manually set lambda_np: ', lambda_np)
                print(f'manual Lambda_np: {man}')
    
    if prior_bool == True:
        lambda_wp, tau_tr, mus = estimate_lambda_wp(select_edge_counts_all, Q, select_lambda_range, prior_matrix)
        # lambda_wp = 0.076
    else:
        lambda_wp = 0

    if verbose:
        print('lambda_np: ', lambda_np)
        print('lambda_wp: ', lambda_wp, '\n')



    # GRAPH OPTIMIZATION WITH FOUND LAMBDAS
    precision_matrix, edge_counts, density = optimize_graph(data, prior_matrix, lambda_np, lambda_wp)

    if verbose:
        print('Number of edges of inferred network (lower triangular): ', edge_counts)
        print('Density: ', density)

    if plot == True:
        scalar_edges = np.sum(edge_counts_all, axis=(0, 1))
        scalar_select_edges = np.sum(select_edge_counts_all, axis=(0, 1))

        if False: # PLOTTING THE TOTAL + THE SELECT RANGE
            # create a 1 x 2 multiplot. on the left, plot both scalar aedes and scalar_select edges. On the right, just scalar_select_edges
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(lambda_range, scalar_edges, color='grey', alpha = 0.5)
            plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
            plt.title(f'#edges vs lambda for {run_type} data,p={p},n={n}')
            plt.xlabel('Lambda')
            plt.ylabel('Number of edges')
            plt.ylim(0, 8000)
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
        if True: # PLOTTING JUST THE TOTAL (WITH RED)
            plt.figure(figsize=(8, 6), dpi=300)
            plt.scatter(lambda_range, scalar_edges, color='grey', alpha = 0.5)
            plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
            plt.title(rf'Edge Counts vs $\lambda$')
            plt.xlabel(r'$ \lambda$', fontsize=15)
            plt.ylabel('Edge Counts', fontsize=12)
            plt.ylim(0, 8000)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
    
    if run_type == 'SYNTHETIC':
        # RECONSTRUCTION
        evaluation_metrics = evaluate_reconstruction(adj_matrix, precision_matrix)
        if verbose:
            print('Evaluation metrics: ', evaluation_metrics)
            print('\n\n\n')

        return precision_matrix, edge_counts, density, lambda_np, lambda_wp, evaluation_metrics
    
    return precision_matrix, edge_counts, density, lambda_np, lambda_wp
    




rank=1
size=1
# ################################################# SYNTHETIC PART #################################################
    
if False:
    ################################################# VARYING SAMPLE SIZE for fixed network #################################################
    p = 137
    n_values = [750] # [50, 100, 200, 400, 750, 1000, 2000]
    b_perc = 0.6
    b = [int(b_perc * n) for n in n_values]   # size of sub-samples
    Q = 1200          # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.5
    granularity = 100
    lambda_range = np.linspace(lowerbound, upperbound, granularity)

    fp_fn = 0.0
    skew = 0
    density = 0.03
    seed = 42

    evalu = {}
    for n,b in zip(n_values, b):
        print(f'NETWORK SIZE for samples: {n} and variables: {p} and b: {b}')
        filename_edges = f'Networks/net_results/synthetic_cmsALL_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{granularity}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{density}_s{seed}.pkl'
        with open(filename_edges, 'rb') as f:
            synth_edge_counts_all = pickle.load(f)
        # divide each value in edge_counts_all by 2*Q
        synth_edge_counts_all = synth_edge_counts_all / (2 * Q)  # EDGE_DIVIDER

        # filename_prior = f'Networks/net_results/prior_mat_synthetic_cmsALL_{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{granularity}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{density}_s{seed}.pkl'
        # with open(filename_prior, 'rb') as f:
        #     synth_prior_matrix = pickle.load(f)

        # # Generate synthetic data and prior matrix
        synth_data, synth_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n, skew=skew, fp_fn_chance=fp_fn, density=density, seed=seed)
        # synth_prior_matrix = synth_prior_matrix * 0

        _, _, _, _, _, temp_evalu = analysis(synth_data, synth_prior_matrix, p, n, Q, lambda_range, lowerbound, upperbound, granularity, 
        synth_edge_counts_all, prior_bool=True, adj_matrix=synth_adj_matrix, run_type='SYNTHETIC', plot=False)

        evalu[n] = temp_evalu

    if False:
        # make plot of evaluation metrics vs n
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(n_values, [evalu[n]['f1_score'] for n in n_values], color='red', alpha=0.8)
        plt.scatter(n_values, [evalu[n]['f1_score'] for n in n_values], color='red', alpha=0.8)
        plt.title(f'F1-Score vs N for synthetic data')
        plt.xlabel('Sample Size N', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.ylim(0.3, 1.1)
        plt.grid()
        ax = plt.gca()
        ax.grid(alpha=0.2)

        plt.subplot(1, 2, 2)
        plt.plot(n_values, [evalu[n]['recall'] for n in n_values], color='red', alpha=0.8)
        plt.scatter(n_values, [evalu[n]['recall'] for n in n_values], color='red', alpha=0.8)
        plt.title(f'Recall vs N for synthetic data', fontsize=12)
        plt.xlabel('Sample Size N', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.ylim(0.3, 1.1)
        plt.grid()
        ax = plt.gca()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


if False:
    # COMPLETE SWEEP
    # Parameter arrays
    p_values = [100, 200, 400]
    n_values = [50, 100, 250, 500, 1000, 2000]
    fp_fn_values = [0.0, 0.25, 0.5, 0.75, 1]
    seed_values = [1, 2, 3, 42]
    dens_values = [0.03, 0.04]


    # Fixed parameters
    Q = 2000
    llo = 0.01
    lhi = 0.5
    lamlen = 100
    b_perc = 0.6
    skew = 0

    lambda_range = np.linspace(llo, lhi, lamlen)

    # Initialize a dictionary to hold f1 scores for averaging
    f1_scores = {}
    recall_scores = {}

    missing_combinations = []
    missing_counts = defaultdict(lambda: defaultdict(int)) 

    # Loop over each parameter combination
    for n in tqdm(n_values):
        for p in p_values:
            for fp_fn in fp_fn_values:
                for seed in seed_values:
                    for dens in dens_values:
                        # Calculate the size of sub-samples (b)
                        b = int(b_perc * n)
                        
                        # Construct filename for edge counts
                        filename_edges = f'Networks/net_results/net_results_sweep/net_results/synthetic_cmsALL_edge_counts_all_pnQ{p}_{n}_{Q}_{llo}_{lhi}_ll{lamlen}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{dens}_s{seed}.pkl'
                        
                        # Check if file exists before trying to open it
                        if not os.path.isfile(filename_edges):
                            missing_combination = f"p={p}, n={n}, fp_fn={fp_fn}, seed={seed}"
                            missing_combinations.append(missing_combination)

                            # Increment missing counts
                            missing_counts['p'][p] += 1
                            missing_counts['n'][n] += 1
                            missing_counts['fp_fn'][fp_fn] += 1
                            missing_counts['seed'][seed] += 1

                            continue  # Skip this file and go to the next one
                        

                        # Load the edge counts
                        with open(filename_edges, 'rb') as f:
                            synth_edge_counts_all = pickle.load(f)
                        
                        # Process the edge counts (your specific processing logic here)
                        synth_edge_counts_all = synth_edge_counts_all / (2 * Q)

                        synth_data, synth_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n, skew=skew, fp_fn_chance=fp_fn, density=dens, seed=seed)
                        
                        if fp_fn == 1:
                            synth_prior_matrix = synth_prior_matrix * 0
                        # Assuming you have a way to calculate or retrieve temp_evalu, focusing on 'f1_score'
                        # For example:
                        _, _, _, _, _, temp_evalu = analysis(synth_data, synth_prior_matrix, p, n, Q, lambda_range, llo, lhi, lamlen, 
                        synth_edge_counts_all, prior_bool=True, adj_matrix=synth_adj_matrix, run_type='SYNTHETIC', plot=False, verbose = False)
                        
                        # Extract 'f1_score' and add to f1_scores dictionary
                        param_key = (p, n, fp_fn, seed, dens)
                        f1_scores[param_key] = temp_evalu['f1_score']
                        recall_scores[param_key] = temp_evalu['recall']

    # save to file
    with open('Networks/net_results/net_results_sweep/f1_scores.pkl', 'wb') as f:
        pickle.dump(f1_scores, f)

    with open('Networks/net_results/net_results_sweep/recall_scores.pkl', 'wb') as f:
        pickle.dump(recall_scores, f)


    average_f1_scores = {}
    average_recall_scores = {}

    for p in p_values:
        for n in n_values:
            for fp_fn in fp_fn_values:
                f1_scores_for_average = []
                recall_scores_for_average = []

                # Collecting all f1 scores for the specific (p, n, fp_fn) across seeds and densities
                for seed in seed_values:
                    for dens in dens_values:
                        key = (p, n, fp_fn, seed, dens)  # Including dens in the key
                        if key in f1_scores:  # Check if the score exists
                            f1_scores_for_average.append(f1_scores[key])
                            recall_scores_for_average.append(recall_scores[key])

                # Calculating the average if there are scores available
                if f1_scores_for_average:  # Check if there are any scores to average
                    average_f1_scores[(p, n, fp_fn)] = sum(f1_scores_for_average) / len(f1_scores_for_average)
                    average_recall_scores[(p, n, fp_fn)] = sum(recall_scores_for_average) / len(recall_scores_for_average)
                else:
                    # Handle case where there are no scores (all data for this combination is missing)
                    average_f1_scores[(p, n, fp_fn)] = None  # or some other indicator of missing data
                    average_recall_scores[(p, n, fp_fn)] = None  # or some other indicator of missing data

    # Now, average_f1_scores will have an average if there's at least one score, or None if all are missing

    # save to file
    with open('Networks/net_results/net_results_sweep/average_f1_scores.pkl', 'wb') as f:
        pickle.dump(average_f1_scores, f)
    
    with open('Networks/net_results/net_results_sweep/average_recall_scores.pkl', 'wb') as f:
        pickle.dump(average_recall_scores, f)


    # write missing combinations to file
    with open('Networks/net_results/net_results_sweep/missing_combinations.txt', 'w') as f:
        for combination in missing_combinations:
            f.write(combination + '\n')

    # Identify parameters that are missing in all cases
    total_combinations = len(p_values) * len(n_values) * len(fp_fn_values) * len(seed_values)
    consistently_missing_params = {param: val for param, counts in missing_counts.items() for val, count in counts.items() if count == total_combinations}

    # # Print or log the results
    # print("Missing parameter combinations:", missing_combinations)
    # print("Consistently missing parameters:", consistently_missing_params)




    # Load average f1 and recall scores from file
    with open('Networks/net_results/net_results_sweep/average_f1_scores.pkl', 'rb') as f:
        average_f1_scores = pickle.load(f)

    with open('Networks/net_results/net_results_sweep/average_recall_scores.pkl', 'rb') as f:
        average_recall_scores = pickle.load(f)

    # Define parameter values from the provided data
    p_values = [100, 200, 400]
    n_values = [50, 100, 250, 500, 1000, 2000]
    fp_fn_values = [0, 0.25, 0.5, 0.75, 1]

    # Create the plots, 3 rows for each p value and 2 columns for F1 and Recall
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey='row')

    for i, p in enumerate(p_values):
        for fp_fn in fp_fn_values:
            # Extracting f1 and recall scores for each n value for the current p and fp_fn setting
            f1_scores = [average_f1_scores.get((p, n, fp_fn)) for n in n_values]
            recall_scores = [average_recall_scores.get((p, n, fp_fn)) for n in n_values]
            
            # Plotting the lines for the current fp_fn setting in F1 and Recall plots
            axes[i, 0].plot(n_values, f1_scores, label=f'Prior Overlap={(1 - fp_fn) * 100}%')
            axes[i, 0].scatter(n_values, f1_scores)  # adding points to indicate actual F1 values
            
            axes[i, 1].plot(n_values, recall_scores, label=f'Prior Overlap={(1 - fp_fn) * 100}%')
            axes[i, 1].scatter(n_values, recall_scores)  # adding points to indicate actual Recall values
            
            # Setting titles and labels
            if i == 0:
                axes[i, 0].set_title('Average F1 Score')
                axes[i, 1].set_title('Average Recall Score')

            axes[i, 0].set_ylabel(f'P = {p}', fontsize=12)
            # make x-xais log2 scale
            axes[i, 0].set_xscale('log', basex=2)
            # make the x labels the exponents of 2
            axes[i, 0].set_xticks(n_values)
            axes[i, 0].set_xticklabels(n_values, fontsize=12)
            
            # axes[i, 1].set_ylabel(f'p = {p}')

    axes[-1, 0].set_xlabel('Sample Size N', fontsize=12)
    axes[-1, 1].set_xlabel('Sample Size N', fontsize=12)
    axes[0, 0].legend(loc='upper right')
    # axes[0, 1].legend(loc='upper right')

    # remove 'fp_fn = 1' from legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = handles[:-1]
    labels = labels[:-1]
    axes[0, 0].legend(handles, labels, loc='upper left')

    # Adjusting layout
    plt.tight_layout()
    # set grid for all subplots
    for ax in axes.flatten():
        ax.grid(alpha=0.3)
    plt.show()






################################################## OMICS DATA PART #################################################
if True:
    for o_t in ['p', 't']:
        for cms in ['cms123', 'cmsALL']:
            # for cms_type in ['cmsALL', 'cms123']:
            #     for omics_type in ['t', 'p']:
            # Parameters
            p = 154
            b_perc = 0.6
            n = 1337             # nnot actual samples, just filename requirements
            Q = 2000             # number of sub-samples

            lowerbound = 0.01
            upperbound = 0.9
            granularity = 500 

            fp_fn = 0
            skew = 0
            density = 0.03
            seed = 42

            # o_t =  't' # omics_type # commented out for loop
            # cms = 'cmsALL' # cms_type # commented out for loop
            # end_slice = 30


            if o_t == 'p':
                prior_bool = True
                omics_type = 'proteomics'
            elif o_t == 't':
                prior_bool = True
                omics_type = 'transcriptomics'

            # Load omics edge counts
            file_ = f'Networks/net_results/{omics_type}_{cms}_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_ll{granularity}_b{b_perc}_fpfn{fp_fn}_skew{skew}_dens{density}_s{seed}.pkl'

            with open(file_, 'rb') as f:
                omics_edge_counts_all = pickle.load(f)

            # divide each value in edge_counts_all by 2*Q
            omics_edge_counts_all = omics_edge_counts_all / (2 * Q)



            # Load Omics Data
            cms_filename = f'Diffusion/data/{omics_type}_for_pig_{cms}.csv'
            cms_filename = 'Diffusion/data/transcriptomics_for_pig_ALL.csv'
            cms_data = pd.read_csv(f'Diffusion/data/{omics_type}_for_pig_{cms}.csv', index_col=0)

            cms_array = cms_data.values



            # LOad Omics Prior Matrix
            if prior_bool == True:
                cms_omics_prior = pd.read_csv('Diffusion/data/RPPA_prior_adj2.csv', index_col=0)
            else:
                cms_omics_prior = pd.read_csv('Diffusion/data/RPPA_prior_adj2.csv', index_col=0)
                #only keep columns / rows that are in the omics data
                cms_omics_prior = cms_omics_prior[cms_data.columns]
                cms_omics_prior = cms_omics_prior.reindex(index=cms_data.columns)
                cms_omics_prior = cms_omics_prior * 0

            cms_omics_prior_matrix = cms_omics_prior.values
            # # Check if there are any non-zero values in the prior matrix
            # print(f'edges in prior: {np.sum(cms_omics_prior_matrix != 0) / 2}')

            # # SYNSUSSYSNSS ##############################################

            # p = 200
            # n = 500
            # fp_fn = 0.0
            # seed = 42
            # dens = 0.03


            # # Fixed parameters
            # Q = 2000
            # lowerbound = 0.01
            # upperbound = 0.5
            # granularity = 100
            # b_perc = 0.6
            # skew = 0
            # filename_edges = 'Networks/net_results/net_results_sweep/net_results/synthetic_cmsALL_edge_counts_all_pnQ200_500_2000_0.01_0.5_ll100_b0.6_fpfn0.25_skew0_dens0.04_s3.pkl'

            # with open(filename_edges, 'rb') as f:
            #         omics_edge_counts_all = pickle.load(f)
                        
            # # Process the edge counts (your specific processing logic here)
            # omics_edge_counts_all = omics_edge_counts_all / (2 * Q)

            # cms_array, cms_omics_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n, skew=skew, fp_fn_chance=fp_fn, density=dens, seed=seed)

            # #### YSNSYNSYHYY END

            p = cms_array.shape[1]
            n = cms_array.shape[0]
            b = int(0.6 * n)

            # scale and center 
            cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(axis=0)


            print(f'{str.upper(omics_type)}, {cms} RESULTS\n-------------------\n')


            print(f'Number of samples: {n}')
            print(f'Number of sub-samples: {Q}')
            print(f'Number of variables: {p}\n')

            # print(f'Granularity of sliced lambda range: {new_granularity}')

            # # RUN ANALYSIS for multiple END SLICES
            if True:
                densities = []
                np_lams = []
                wp_lams = []
                no_end_slices = 300
                slicer_range = range(200, no_end_slices)
                x_axis = []
                i = 0
                for end_slice in slicer_range:
                    i += 1
                    sliced_omics_edge_counts_all = omics_edge_counts_all[:,:,:-end_slice]

                    # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
                    new_granularity = sliced_omics_edge_counts_all.shape[2]
                    new_upperbound = lowerbound + (upperbound - lowerbound) * (new_granularity - 1) / (granularity - 1)
                    x_axis.append(new_upperbound)

                    lambda_range = np.linspace(lowerbound, new_upperbound, new_granularity)
                    kpa = 0                                                                                                        # HERE
                    precision_mat, edge_counts, density, lambda_np, lambda_wp = analysis(cms_array, cms_omics_prior_matrix, p, n, Q, lambda_range, 
                                lowerbound, new_upperbound, new_granularity, sliced_omics_edge_counts_all, prior_bool, run_type='OMICS', kneepoint_adder=kpa, plot=False)

                    print(i, new_upperbound, o_t, cms)
                    print(f'lambda_np: {lambda_np}, lambda_wp: {lambda_wp}, density: {density}')
                    densities.append(density)
                    np_lams.append(lambda_np)
                    wp_lams.append(lambda_wp)
                
                # write densities to file
                with open(f'Networks/net_results/Sendslice_densities_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.pkl', 'wb') as f:
                    pickle.dump(densities, f)
                # write np_lams to file
                with open(f'Networks/net_results/Sendslice_np_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.pkl', 'wb') as f:
                    pickle.dump(np_lams, f)
                # write wp_lams to file
                with open(f'Networks/net_results/Sendslice_wp_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.pkl', 'wb') as f:
                    pickle.dump(wp_lams, f)

                # # load np_lams from file
                # with open(f'Networks/net_results/endslice_np_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.pkl', 'rb') as f:
                #     np_lams = pickle.load(f)

                # # load wp_lams from file
                # with open(f'Networks/net_results/endslice_wp_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.pkl', 'rb') as f:
                #     wp_lams = pickle.load(f)

                # # plot both np_lams and wp_lams against end slice value
                # plt.figure(figsize=(12, 5))
                # plt.plot(x_axis, np_lams, color='red', alpha=0.8, label=r'$\lambda_{np}$')
                # plt.scatter(x_axis, np_lams, color='red', alpha=0.8)
                # plt.plot(x_axis, wp_lams, color='blue', alpha=0.8, label=r'$\lambda_{wp}$')
                # plt.scatter(x_axis, wp_lams, color='blue', alpha=0.8)
                # plt.title(f'$\lambda_np$ and $\lambda_wp$ vs end slice value for {omics_type} data, Q = {Q}')
                # plt.xlabel('End slice value', fontsize=12)
                # plt.ylabel(r'$\lambda$', fontsize=12)
                # # plt.ylim(0.0, 0.5)

                plt.show()

                # # load densities from file
                with open(f'Networks/net_results/endslice_densities_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.pkl', 'rb') as f:
                    densities = pickle.load(f)
                    
                # plot density against end slice value
                plt.figure(figsize=(12, 5))
                plt.plot(x_axis, densities, color='red', alpha=0.8)
                plt.scatter(x_axis, densities, color='red', alpha=0.8)
                plt.title(f'Density vs end slice value for {omics_type} data, Q = {Q}')
                plt.xlabel('End slice value', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                # plt.ylim(0.0, 0.5)
                plt.grid()
                ax = plt.gca()
                ax.grid(alpha=0.2)
                plt.tight_layout()

                # save plot image to file
                plt.savefig(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Pictures/Pics_11_12_23/density_vs_end_slice_{omics_type}_{cms}_Q{Q}_prior{prior_bool}.png')

                # plt.show()
            
            else:
                sliced_omics_edge_counts_all = omics_edge_counts_all[:,:,:-end_slice]

                # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
                new_granularity = sliced_omics_edge_counts_all.shape[2]
                new_upperbound = lowerbound + (upperbound - lowerbound) * (new_granularity - 1) / (granularity - 1)
                lambda_range = np.linspace(lowerbound, new_upperbound, new_granularity)

                kpa = 0                                                                                                        # HERE
                precision_mat, edge_counts, density, lambda_np, lambda_wp = analysis(cms_array, cms_omics_prior_matrix, p, n, Q, lambda_range, 
                            lowerbound, new_upperbound, new_granularity, sliced_omics_edge_counts_all, prior_bool, run_type='OMICS', kneepoint_adder=kpa, plot=False)


            # # get adjacency from precision matrix
            # adj_matrix = (np.abs(precision_mat) > 1e-5).astype(int)
            # # assign columns and indices of prior matrix to adj_matrix
            # adj_matrix = pd.DataFrame(adj_matrix, index=cms_data.columns, columns=cms_data.columns)

            # # WRITE ADJACAENCY MATRIX TO FILE
            # # # save adjacency matrix
            # # adj_matrix.to_csv(f'Networks/net_results/inferred_adjacencies/{omics_type}_{cms}_adj_matrix_p{p}_kpa{kpa}_lowenddensity.csv')









            # # # draw the network
            # # G = nx.from_pandas_adjacency(cms_omics_prior)
            # # nx.draw(G, with_labels=True)
            # # plt.title(f'Network for {omics_type} data')
            # # plt.show()

            # # #plot the degree distribution
            # G = nx.from_pandas_adjacency(adj_matrix)
            # degrees = [G.degree(n) for n in G.nodes()]

            # # get layout and draw the network
            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, with_labels=True)
            # plt.title(f'Network for {omics_type} data')
            # plt.show()


            # # plt.hist(degrees, bins=20)
            # # plt.title(f'Degree distribution for {omics_type} data')
            # # plt.xlabel('Degree')
            # # plt.ylabel('Frequency')
            # # plt.show()

            # highest_degrees_indices = list(np.argsort(degrees)[-20:])
            # nodes_with_highest_degrees = [list(G.nodes())[i] for i in highest_degrees_indices]

            # print(f'Highest degrees: {np.sort(degrees)[-20:]}')
            # print(f'Nodes with highest degrees: {nodes_with_highest_degrees}')

            # # Print the degree of 'TP53'
            # print(f'Degree of TP53: {G.degree("TP53")}')



            # # # LOG - LOG SCALE   
            # # # Count the frequency of each degree
            # # degree_counts = Counter(degrees)
            # # degrees, counts = zip(*degree_counts.items())
            # # # Scatter plot
            # # plt.scatter(degrees, counts)
            # # # Set both axes to logarithmic scale
            # # plt.xscale('log')
            # # plt.yscale('log')
            # # # Set the labels and title
            # # plt.xlabel('Degree')
            # # plt.ylabel('Frequency')
            # # plt.title(f'Log-Log Scatter Plot of Degree Distribution for {omics_type} data')
            # # # Show the plot
            # # plt.show()






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