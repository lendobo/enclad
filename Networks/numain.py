from estimate_lambdas import estimate_lambda_np, estimate_lambda_wp, find_all_knee_points
from piglasso import QJSweeper
from evaluation_of_graph import optimize_graph, evaluate_reconstruction

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import scipy.stats as stats
from collections import Counter

from collections import defaultdict
import os
import sys

# from tqdm import tqdm
import tqdm
from multiprocessing import Pool
from itertools import product

# original_stdout = sys.stdout
# sys.stdout = open('Networks/net_results/Piglasso_Logs.txt', 'w')


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
        man_param=False,
        adj_matrix=None, 
        run_type='SYNTHETIC',
        plot=False,
        verbose=False):

    # KNEE POINTS
    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)

    l_lo = left_knee_point_index     # Set it at knee-point index -1 
    if run_type == 'SYNTHETIC':
        l_hi = right_knee_point_index # set at knee-point index
    else:
        l_hi = right_knee_point_index  
        if verbose:

            print(f'right_knee_point_index: {right_knee_point_index}')
        # print(f'selected l_hi: {lambda_range[l_hi]}')
    

    select_lambda_range = lambda_range[l_lo:l_hi]
    select_edge_counts_all = edge_counts_all[:, :, l_lo:l_hi]

    if verbose:
        print(f'complete lambda range: {lowerbound, upperbound}')                                                              # HERE
        print(f'Selected lambda range: {select_lambda_range[0]} - {select_lambda_range[-1]} \n')

    # LAMBDAS
    lambda_np, theta_mat = estimate_lambda_np(select_edge_counts_all, Q, select_lambda_range)
    man = man_param
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
        tau_tr = 1e+5

    if verbose:
        print('lambda_np: ', lambda_np)
        print('lambda_wp: ', lambda_wp, '\n')

    # GRAPH OPTIMIZATION WITH FOUND LAMBDAS
    precision_matrix, edge_counts, density = optimize_graph(data, prior_matrix, lambda_np, lambda_wp, verbose=verbose)

    if verbose:
        print('Number of edges of inferred network (lower triangular): ', edge_counts)
        print('Density: ', density)

    if plot == True:
        scalar_edges = np.sum(edge_counts_all, axis=(0, 1)) / (2 * Q)
        scalar_select_edges = np.sum(select_edge_counts_all, axis=(0, 1)) / (2 * Q)

        if False: # PLOTTING THE TOTAL + THE SELECT RANGE
            # create a 1 x 2 multiplot. on the left, plot both scalar aedes and scalar_select edges. On the right, just scalar_select_edges
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(lambda_range, scalar_edges, color='grey', alpha = 0.5)
            plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
            plt.title(f'#edges vs lambda for {run_type} data,p={p},n={n}')
            plt.xlabel('Lambda')
            plt.ylabel('Number of edges')
            # plt.ylim(0, 8000)
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
        if True: # PLOTTING JUST THE TOTAL (WITHOUT RED)
            plt.figure(figsize=(8, 6), dpi=300)
            plt.scatter(lambda_range, scalar_edges, color='grey', alpha = 0.5)
            plt.scatter(select_lambda_range, scalar_select_edges, color='red', alpha=0.8)
            plt.title(rf'Edge Counts vs $\lambda$')
            plt.xlabel(r'$ \lambda$', fontsize=15)
            plt.ylabel('Edge Counts', fontsize=12)
            # plt.ylim(0, 8000)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
    
    if run_type == 'SYNTHETIC':
        # RECONSTRUCTION
        evaluation_metrics = evaluate_reconstruction(adj_matrix, precision_matrix)
        if verbose:
            print('Evaluation metrics: ', evaluation_metrics)
            print('\n\n\n')

        return precision_matrix, edge_counts, density, lambda_np, lambda_wp, evaluation_metrics, tau_tr
    
    return precision_matrix, edge_counts, density, lambda_np, lambda_wp, tau_tr
    




rank=1
size=1
# ################################################# SYNTHETIC PART #################################################
if True:
    run = False
    # COMPLETE SWEEP
    # code should compare: increasing B_perc and effect on performance at low sample size vs high sample size
    # for 250 samples, which combination of b_perc and manual lambda (T, F) and fp_fn is best?
    # Then we have to assess our fp_fn from the tau parameter
        # How does overlap (x axis) correlate with tau (y axis)? NEXT ONE 
    # Finally, make a call on instability G
    # Parameter arrays

    # INCREASE P as well

    # CONCLUSIONS
    # for low sample size (250), b_perc between 0.7 and 0.8 is best. Since we also have sampl size of 350, let's assume 0.7 is optimal
    # manual lambda vs inferred?

    p_values = [150]
    n_values = [50, 100, 300, 500, 700, 900, 1100] # [75, 250, 500, 750, 1000] 
    b_perc_values = [0.6, 0.65, 0.7]
    fp_fn_values = [0.0, 0.6, 0.8, 1] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1] # Try without 0.1 in plotting
    seed_values = list(range(1, 31))
    dens_values = [0.05]
    man_values = [False, True]


    # Fixed parameters
    Q = 1000
    llo = 0.01
    lhi = 0.5
    lamlen = 100
    skew = 0
    prior_bool = True

    lambda_range = np.linspace(llo, lhi, lamlen)

    # Initialize a dictionary to hold f1 scores for averaging
    f1_scores = {}
    recall_scores = {}

    if "SLURM_JOB_ID" not in os.environ:
        dir_prefix = 'Networks/'
    else:
        dir_prefix = ''

    if run == True:
        def worker_function(params):
            p, n, b_perc, fp_fn, seed, dens, man = params

            # Your fixed parameters
            Q = 1000
            llo = 0.01
            lhi = 0.5
            lamlen = 100
            skew = 0
            prior_bool = True
            lambda_range = np.linspace(llo, lhi, lamlen)

            # Calculate the size of sub-samples (b)
            b = int(b_perc * n)
            
            # Construct filename for edge counts
            filename_edges = f'{dir_prefix}net_results/synthetic_cmsALL_edge_counts_all_pnQ{p}_{n}_{Q}_{llo}_{lhi}_ll{lamlen}_b{b_perc}_fpfn0.0_skew0_dens{dens}_s{seed}.pkl'
            param_key = (p, n, b_perc, fp_fn, seed, dens, str(man))

            if not os.path.isfile(filename_edges):
                return None  # File does not exist

            with open(filename_edges, 'rb') as f:
                synth_edge_counts_all = pickle.load(f)

            # Process the edge counts
            synth_edge_counts_all = synth_edge_counts_all #  / (2 * Q)

            synth_data, synth_prior_matrix, synth_adj_matrix = QJSweeper.generate_synth_data(p, n, skew=skew, fp_fn_chance=fp_fn, density=dens, seed=seed)

            overlap = np.sum((synth_prior_matrix == 1) & (synth_adj_matrix == 1)) / (np.sum(synth_prior_matrix == 1))


            if fp_fn == 1:
                synth_prior_matrix = synth_prior_matrix * 0
                prior_bool = False

            # Run your analysis
            _, _, _, lambda_np, lambda_wp, temp_evalu, tau_tr = analysis(synth_data, synth_prior_matrix, p, n, Q, lambda_range, llo, lhi, lamlen, 
                                                synth_edge_counts_all, prior_bool=prior_bool, man_param=man, adj_matrix=synth_adj_matrix, run_type='SYNTHETIC', plot=False, verbose=False)

            # print('F1 SCORE', temp_evalu['f1_score'])
            # print(f'NP: {lambda_np}, WP: {lambda_wp}')
            # print('tau_tr', tau_tr)
            # print(f'overlap: {overlap}')


            return {
                'param_key': param_key,
                'f1_score': temp_evalu['f1_score'],
                'precision': temp_evalu['precision'],
                'recall': temp_evalu['recall'],
                'jaccard_similarity': temp_evalu['jaccard_similarity'],
                'overlap': overlap,
                'tau_tr': tau_tr,
                'lambda_np': lambda_np,
                'lambda_wp': lambda_wp
            }
        
        def update_progress(*a):
            pbar.update()


        if __name__ == "__main__":
            parameter_combinations = list(product(p_values, n_values, b_perc_values, fp_fn_values, seed_values, dens_values, man_values))

            with Pool() as pool:
                pbar = tqdm.tqdm(total=len(parameter_combinations))
                results = [pool.apply_async(worker_function, args=(params,), callback=update_progress) for params in parameter_combinations]
                
                # Close the pool and wait for each task to complete
                pool.close()
                pool.join()
                pbar.close()

            # Extract the results from each async result object
            results = [res.get() for res in results]

            # Organize results
            organized_results = {result['param_key']: {'f1_score': result['f1_score'], 'recall': result['recall'], 'overlap': result['overlap'], 'tau_tr': result['tau_tr']} 
                                for result in results if result is not None}

            # save to file
            with open(f'{dir_prefix}net_results/net_results_sweep/organized_SWEEP_results_n{len(n_values)}_withjaccetc.pkl', 'wb') as f:
                pickle.dump(organized_results, f)

            print("Organized results saved.")

    post_process = False
    if post_process == True:
        # Load the organized results
        with open(f'{dir_prefix}net_results/net_results_sweep/organized_SWEEP_results_n{len(n_values)}.pkl', 'rb') as f:
            organized_results = pickle.load(f)

        # Initialize dictionaries for average scores and SDs
        average_f1_scores = {}
        SD_f1_scores = {}
        average_recall_scores = {}
        SD_recall_scores = {}

        average_overlap_scores = {}

        f1_counts = {}

        # Loop over parameter combinations
        for p in p_values:
            for n in n_values:
                for b_perc in b_perc_values:
                    for fp_fn in fp_fn_values:
                        for man in [str(man) for man in man_values]:
                            f1_scores_for_average = []
                            recall_scores_for_average = []
                            overlap_scores_for_average = []

                            # New key without seed and dens
                            new_key = (p, n, b_perc, fp_fn, man)
                            f1_counts[new_key] = 0

                            # Loop over seeds and densities
                            for seed in seed_values:
                                for dens in dens_values:
                                    key = (p, n, b_perc, fp_fn, seed, dens, man)
                                    result = organized_results.get(key)
                                    if result:  # Check if the result exists
                                        f1_scores_for_average.append(result['f1_score'])
                                        recall_scores_for_average.append(result['recall'])
                                        overlap_scores_for_average.append(result['overlap'])

                                        # Increment the f1 count
                                        f1_counts[new_key] += 1



                            # Calculating the average and SD
                            if f1_scores_for_average:
                                average_f1_scores[new_key] = np.mean(f1_scores_for_average)
                                SD_f1_scores[new_key] = np.std(f1_scores_for_average)
                                average_recall_scores[new_key] = np.mean(recall_scores_for_average)
                                SD_recall_scores[new_key] = np.std(recall_scores_for_average)
                                average_overlap_scores[new_key] = np.mean(overlap_scores_for_average)
                            else:
                                # Handle missing data
                                average_f1_scores[new_key] = None
                                SD_f1_scores[new_key] = None
                                average_recall_scores[new_key] = None
                                SD_recall_scores[new_key] = None
                                average_overlap_scores[new_key] = None



        # Now, average_f1_scores will have an average if there's at least one score, or None if all are missing

        # save to file
        with open(f'{dir_prefix}net_results/net_results_sweep/average_f1_scores.pkl', 'wb') as f:
            pickle.dump(average_f1_scores, f)

        with open(f'{dir_prefix}net_results/net_results_sweep/SD_f1_scores.pkl', 'wb') as f:
            pickle.dump(SD_f1_scores, f)
        
        with open(f'{dir_prefix}net_results/net_results_sweep/average_recall_scores.pkl', 'wb') as f:
            pickle.dump(average_recall_scores, f)

        with open(f'{dir_prefix}net_results/net_results_sweep/SD_recall_scores.pkl', 'wb') as f:
            pickle.dump(SD_recall_scores, f)

        with open(f'{dir_prefix}net_results/net_results_sweep/average_overlap_scores.pkl', 'wb') as f:
            pickle.dump(average_overlap_scores, f)

        # # # # # 
        # write f1 counts to a txt file
        with open(f'{dir_prefix}net_results/net_results_sweep/f1_counts.txt', 'w') as f:
            for item in f1_counts.items():
                f.write(f'{item}\n')


    # UNCOMMENT TO LOAD F1 scores averages FROM FILE
    # Load average f1 and recall scores from file
    with open(f'{dir_prefix}net_results/net_results_sweep/average_f1_scores.pkl', 'rb') as f:
        average_f1_scores = pickle.load(f)

    with open(f'{dir_prefix}net_results/net_results_sweep/average_recall_scores.pkl', 'rb') as f:
        average_recall_scores = pickle.load(f)

    # load SDs
    with open(f'{dir_prefix}net_results/net_results_sweep/SD_f1_scores.pkl', 'rb') as f:
        SD_f1_scores = pickle.load(f)

    with open(f'{dir_prefix}net_results/net_results_sweep/SD_recall_scores.pkl', 'rb') as f:
        SD_recall_scores = pickle.load(f)

    # load overlap scores 
    with open (f'{dir_prefix}net_results/net_results_sweep/average_overlap_scores.pkl', 'rb') as f:
        average_overlap_scores = pickle.load(f)
    


    # PLOTTING
    if False: # B_PERC PLOTTING
        n = 700  # Fixed sample size
        p = 150  # Fixed number of variables
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))  # 2x2 subplot

        # Iterate over fp_fn values and plot
        for fp_fn in fp_fn_values:
            for i, man in enumerate(man_values):
                f1_scores = []
                recall_scores = []
                f1_errors = []
                recall_errors = []

                for b_perc in b_perc_values:
                    # Accessing the scores using the correct keys
                    key = (p, n, b_perc, fp_fn, str(man))
                    f1_scores.append(average_f1_scores.get(key))
                    recall_scores.append(average_recall_scores.get(key))
                    f1_errors.append(SD_f1_scores.get(key, 0))  # Default to 0 if no SD available
                    recall_errors.append(SD_recall_scores.get(key, 0))  # Default to 0 if no SD available

                # Plot F1 scores in the first column with error bars
                axes[i, 0].errorbar(b_perc_values, f1_scores, yerr=f1_errors, label=f'overlap={average_overlap_scores[key]}', fmt='-o')
                axes[i, 0].set_title(f'F1 Scores, Manual={man}')
                axes[i, 0].set_xlabel('b_perc')
                axes[i, 0].set_ylabel('F1 Score')
                axes[i, 0].legend(loc='best')
                axes[i, 0].grid(alpha=0.3)

                # Plot Recall scores in the second column with error bars
                axes[i, 1].errorbar(b_perc_values, recall_scores, yerr=recall_errors, label=f'overlap={average_overlap_scores[key]}', fmt='-o')
                axes[i, 1].set_title(f'Recall Scores, Manual={man}')
                axes[i, 1].set_xlabel('b_perc')
                axes[i, 1].set_ylabel('Recall Score')
                axes[i, 1].legend(loc='best')
                axes[i, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    if False:  # N VALUE PLOTTING
        def reversed_colormap(cmap_name):
            cmap = plt.cm.get_cmap(cmap_name)
            colors = cmap(np.arange(cmap.N))
            colors = np.flipud(colors)
            return mcolors.LinearSegmentedColormap.from_list('reversed_' + cmap_name, colors)

        # Use the custom function to reverse the 'Blues' colormap
        reversed_blues = reversed_colormap('Blues')

        b_perc = 0.65  # Fixed b_perc
        p = 150  # Fixed number of variables
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 15), sharex=True, dpi=150)  # 1x2 subplot
        # set suptitle
        # fig.suptitle(f'F-Scores and Recall for Synthetic Data', fontsize=16)

        man = False  # Only considering False for man values

        # Iterate over fp_fn values and plot
        for fp_fn in fp_fn_values:
            if fp_fn < 1.0:
                exponent = 3
                scaled_fp_fn = fp_fn ** exponent
                color = reversed_blues(scaled_fp_fn)
                alpha = 1
            else:
                color = 'firebrick'
                alpha = 1

            f1_scores = []
            recall_scores = []
            f1_errors = []
            recall_errors = []

            for n in n_values:
                # Accessing the scores using the correct keys
                key = (p, n, b_perc, fp_fn, str(man))
                f1_scores.append(average_f1_scores.get(key))
                recall_scores.append(average_recall_scores.get(key))
                f1_errors.append(SD_f1_scores.get(key, 0))  # Default to 0 if no SD available
                recall_errors.append(SD_recall_scores.get(key, 0))  # Default to 0 if no SD available

            # Plot F1 scores
            axes[0].errorbar(n_values, f1_scores, yerr=f1_errors, fmt='-o', color=color, alpha=alpha, markersize=3, label=f'overlap={round(average_overlap_scores[key], 1)}')
            # axes[0].set_title('F1 Scores', fontsize=14)
            # axes[0].set_xlabel('Sample Size', fontsize=14)
            axes[0].set_ylabel('F-Score', fontsize=12)
            axes[0].legend(loc='best')
            axes[0].grid(alpha=0.15)
            # remove xticks
            axes[0].set_xticks([])

            # Plot Recall scores
            axes[1].errorbar(n_values, recall_scores, yerr=recall_errors, fmt='-o', color=color, alpha=alpha, markersize=3, label=f'overlap={round(average_overlap_scores[key], 1)}')
            # axes[1].set_title('Recall Scores', fontsize=14)
            axes[1].set_xlabel('Sample Size', fontsize=12)
            axes[1].set_ylabel('Recall', fontsize=12)
            # axes[1].legend(loc='best')
            axes[1].grid(alpha=0.15)


            xticks = [0, 100, 300, 500, 700, 900, 1100]
            for ax in axes:  # Apply to both subplots
                ax.set_xticks(xticks)
                ax.set_xlim(0, 1150)

        # plt.tight_layout()
        plt.show()



    if True: # TAU vs OVERLAP PLOTTING
        # load organized results
        with open(f'{dir_prefix}net_results/net_results_sweep/organized_SWEEP_results_n{len(n_values)}.pkl', 'rb') as f:
            organized_results = pickle.load(f)
        # Organize data by 'overlap'
        organized_data = {}
        for key, value in organized_results.items():
            if value['overlap'] == 0.0:
                continue
            overlap = value['overlap']
            tau_tr = value['tau_tr']
            if overlap not in organized_data:
                organized_data[overlap] = []
            organized_data[overlap].append(tau_tr)

        # Calculate mean and standard deviation for each 'overlap'
        overlap_values = []
        mean_tau_tr_values = []
        error_tau_tr_values = []
        for overlap, tau_tr_list in organized_data.items():
            overlap_values.append(overlap)
            mean_tau_tr_values.append(np.mean(tau_tr_list))
            error_tau_tr_values.append(np.std(tau_tr_list))

        # Plotting
        plt.figure(figsize=(7, 4), dpi = 300)
        plt.errorbar(overlap_values, mean_tau_tr_values, yerr=error_tau_tr_values, fmt='o', color='purple', alpha=0.5)
        plt.plot(overlap_values, mean_tau_tr_values, color='purple', alpha=0.5)
        # plt.title(r'Average $\tau^{t_r}$ vs Prior Overlap for Synthetic Data')
        plt.xlabel('Prior Overlap', fontsize=14)
        plt.ylabel(r'$\tau^{t_r}$', fontsize=18)
        plt.grid()
        ax = plt.gca()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()








################################################## OMICS DATA PART #################################################
if False:
    for o_t in ['p', 't']:
        for cms in ['cmsALL', 'cms123']:
            # for cms_type in ['cmsALL', 'cms123']:
            #     for omics_type in ['t', 'p']:
            # Parameters
            p = 154
            b_perc = 0.65
            n = 1337             # nnot actual samples, just filename requirements
            Q = 1000             # number of sub-samples

            lowerbound = 0.01
            upperbound = 1.5
            granularity = 500
            lambda_range = np.linspace(lowerbound, upperbound, granularity)

            fp_fn = 0
            skew = 0
            density = 0.03
            seed = 42

            man = False
            smooth_bool = False
            net_dens = 'low_dens'

            # o_t =  't' # omics_type # commented out for loop
            # cms = 'cmsALL' # cms_type # commented out for loop

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

            if smooth_bool == True:
                # smooth the data
                window_size = 30
                def smooth_data(data, window_size):
                    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

                p = omics_edge_counts_all.shape[0]  # Assuming the shape is (p, p, len(lambda_range))
                smoothed_edge_counts_all = np.zeros((p, p, len(lambda_range) - window_size + 1))

                for i in range(p):
                    for j in range(p):
                        smoothed_edge_counts_all[i, j] = smooth_data(omics_edge_counts_all[i, j], window_size)

                omics_edge_counts_all = smoothed_edge_counts_all
                # # divide each value in edge_counts_all by 2*Q
                # omics_edge_counts_all = omics_edge_counts_all


            # Load Omics Data
            # cms_filename = f'Diffusion/data/{omics_type}_for_pig_{cms}.csv'
            cms_data = pd.read_csv(f'Diffusion/data/{omics_type}_for_pig_{cms}.csv', index_col=0)

            cms_array = cms_data.values

            # LOad Omics Prior Matrix
            if prior_bool == True:
                cms_omics_prior = pd.read_csv('Diffusion/data/RPPA_prior_adj90perc.csv', index_col=0)
                # print density of prior
                complete_g = (p * (p - 1))
                prior_density = np.sum(cms_omics_prior.values) / complete_g
                print(f'prior density: {prior_density}')
            else:
                cms_omics_prior = pd.read_csv('Diffusion/data/RPPA_prior_adj90perc.csv', index_col=0)
                #only keep columns / rows that are in the omics data
                cms_omics_prior = cms_omics_prior[cms_data.columns]
                cms_omics_prior = cms_omics_prior.reindex(index=cms_data.columns)
                cms_omics_prior = cms_omics_prior * 0

            cms_omics_prior_matrix = cms_omics_prior.values * 0.9
            # Check if there are any non-zero values in the prior matrix
            print(f'edges in prior: {np.sum(cms_omics_prior_matrix != 0) / 2}')

            p = cms_array.shape[1]
            n = cms_array.shape[0]
            b = int(0.65 * n)

            # scale and center 
            cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(axis=0)


            print(f'{str.upper(omics_type)}, {cms} RESULTS\n-------------------\n')


            print(f'Number of samples: {n}')
            print(f'Number of sub-samples: {Q}')
            print(f'Number of variables: {p}\n')

            # print(f'Granularity of sliced lambda range: {new_granularity}')

            # # RUN ANALYSIS for multiple END SLICES
            if False:
                densities = []
                np_lams = []
                wp_lams = []
                tau_trs = []
                no_end_slices = 400
                slicer_range = range(200, no_end_slices)
                x_axis = []
                i = 0
                for end_slice in slicer_range:
                    i += 1
                    sliced_omics_edge_counts_all = omics_edge_counts_all[:,:,:-end_slice]

                    # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
                    new_granularity = sliced_omics_edge_counts_all.shape[2]
                    new_upperbound = lowerbound + (upperbound - lowerbound) * (new_granularity - 1) / (granularity - 1)
                    # x_axis.append(new_upperbound)
                    x_axis.append(end_slice)

                    lambda_range = np.linspace(lowerbound, new_upperbound, new_granularity)
                    kpa = 0                                                                                                        # HERE
                    precision_mat, edge_counts, density, lambda_np, lambda_wp, tau_tr = analysis(cms_array, cms_omics_prior_matrix, p, n, Q, lambda_range, 
                                lowerbound, new_upperbound, new_granularity, sliced_omics_edge_counts_all, prior_bool, man_param=man, run_type='OMICS', plot=False, verbose=False)

                    print(i, new_upperbound, o_t, cms)
                    print(f'lambda_np: {lambda_np}, lambda_wp: {lambda_wp}, density: {density}')
                    densities.append(density)
                    np_lams.append(lambda_np)
                    wp_lams.append(lambda_wp)
                    tau_trs.append(tau_tr)
                
                # write densities to file
                with open(f'Networks/net_results/endslice_densities_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'wb') as f:
                    pickle.dump(densities, f)
                # write np_lams to file
                with open(f'Networks/net_results/endslice_np_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'wb') as f:
                    pickle.dump(np_lams, f)
                # write wp_lams to file
                with open(f'Networks/net_results/endslice_wp_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'wb') as f:
                    pickle.dump(wp_lams, f)
                # write tau_trs to file
                with open(f'Networks/net_results/endslice_tau_trs_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'wb') as f:
                    pickle.dump(tau_trs, f)


                # # # 
               
                # Load np_lams and wp_lams from file
                with open(f'Networks/net_results/endslice_np_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'rb') as f:
                    np_lams = pickle.load(f)
                with open(f'Networks/net_results/endslice_wp_lams_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'rb') as f:
                    wp_lams = pickle.load(f)

                # Load tau_trs and densities from file
                with open(f'Networks/net_results/endslice_tau_trs_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'rb') as f:
                    tau_trs = pickle.load(f)
                with open(f'Networks/net_results/endslice_densities_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}.pkl', 'rb') as f:
                    densities = pickle.load(f)

                # Create a figure with 3 subplots
                plt.figure(figsize=(18, 5))  # Adjust the size as needed

                # First subplot for np_lams and wp_lams
                plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
                plt.plot(x_axis, np_lams, color='red', alpha=0.8, label=r'$\lambda_{np}$')
                plt.scatter(x_axis, np_lams, color='red', alpha=0.8)
                plt.plot(x_axis, wp_lams, color='blue', alpha=0.8, label=r'$\lambda_{wp}$')
                plt.scatter(x_axis, wp_lams, color='blue', alpha=0.8)
                plt.title(f'$\lambda_np$ and $\lambda_wp$ vs end slice value for {omics_type} data, Q = {Q}')
                plt.xlabel('End slice value', fontsize=12)
                plt.ylabel(r'$\lambda$', fontsize=12)
                plt.legend()
                plt.grid()

                # Second subplot for tau_trs
                plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
                plt.plot(x_axis, tau_trs, color='purple', alpha=0.65)
                plt.scatter(x_axis, tau_trs, color='purple', alpha=0.65)
                plt.title(fr'$\tau_{{tr}}$ vs end slice value for {omics_type} data, Q = {Q}')
                plt.xlabel('End slice value', fontsize=12)
                plt.ylabel(r'$\tau_{tr}$', fontsize=12)
                plt.grid()

                # Third subplot for densities
                plt.subplot(1, 3, 3) # 1 row, 3 columns, third subplot
                plt.plot(x_axis, densities, color='red', alpha=0.8)
                plt.scatter(x_axis, densities, color='red', alpha=0.8)
                plt.title(f'Density vs end slice value for {omics_type} data, Q = {Q}')
                plt.xlabel('End slice value', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.grid()

                plt.tight_layout()

                # Show the figure

                plt.show()
                # Save the figure to file

                plt.savefig(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Pictures/Pics_11_12_23/multiplot_3col_{omics_type}_{cms}_Q{Q}_prior{prior_bool}_slices{len(slicer_range)}_smoothing{smooth_bool}.png')

            
            else:
                if net_dens == 'high_dens':
                    end_slice = 325
                else:
                    end_slice = 250
                sliced_omics_edge_counts_all = omics_edge_counts_all[:,:,:-end_slice]

                # SETTING LAMBDA DIMENSIONS TO FIT THE DATA
                new_granularity = sliced_omics_edge_counts_all.shape[2]
                new_upperbound = lowerbound + (upperbound - lowerbound) * (new_granularity - 1) / (granularity - 1)
                lambda_range = np.linspace(lowerbound, new_upperbound, new_granularity)

                                                                                                                        # HERE
                precision_mat, edge_counts, density, lambda_np, lambda_wp, tau_tr = analysis(cms_array, cms_omics_prior_matrix, p, n, Q, lambda_range, 
                            lowerbound, new_upperbound, new_granularity, sliced_omics_edge_counts_all, prior_bool, man_param=man, run_type='OMICS', plot=True, verbose=True)

            # print tau_tr value
            print(f'tau_tr: {tau_tr}')

            # get adjacency from precision matrix
            adj_matrix = (np.abs(precision_mat) > 1e-5).astype(int)
            # assign columns and indices of prior matrix to adj_matrix
            adj_matrix = pd.DataFrame(adj_matrix, index=cms_data.columns, columns=cms_data.columns)

            # # WRITE ADJACAENCY MATRIX TO FILE
            # save adjacency matrix
            # adj_matrix.to_csv(f'Networks/net_results/inferred_adjacencies/{omics_type}_{cms}_adj_matrix_p{p}_Lambda_np{not man}_{net_dens}.csv')

            # # compare similarity of adj_matrix and prior matrix using evaluate_reconstruction
            # evaluation_metrics = evaluate_reconstruction(cms_omics_prior_matrix, adj_matrix.values)
            # print(f'Similarity of inferred net to prior: {evaluation_metrics}\n\n')



            # # # draw the network
            G = nx.from_pandas_adjacency(adj_matrix)
            # get number of orphan nodes
            orphan_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
            print(f'Number of orphan nodes: {len(orphan_nodes)}')
            # print names of orphan nodes
            print(f'Names of orphan nodes: {orphan_nodes}\n\n')

            # nx.draw(G, with_labels=True)
            # # plt.title(f'Network for {omics_type} data')
            # # plt.show()

            # # #plot the degree distribution
            # G = nx.from_pandas_adjacency(adj_matrix)
            # degrees = [G.degree(n) for n in G.nodes()]
            # plt.hist(degrees, bins=20)
            # plt.title(f'Degree distribution for {omics_type} data')
            # plt.xlabel('Degree')
            # plt.ylabel('Frequency')
            # plt.show()

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

    proteomics_ALL_net = pd.read_csv(f'Networks/net_results/inferred_adjacencies/proteomics_cmsALL_adj_matrix_p154_Lambda_np{not man}.csv', index_col=0)
    transcriptomics_ALL_net = pd.read_csv(f'Networks/net_results/inferred_adjacencies/transcriptomics_cmsALL_adj_matrix_p154_Lambda_np{not man}.csv', index_col=0)
    proteomics_123_net = pd.read_csv(f'Networks/net_results/inferred_adjacencies/proteomics_cms123_adj_matrix_p154_Lambda_np{not man}.csv', index_col=0)
    transcriptomics_123_net = pd.read_csv(f'Networks/net_results/inferred_adjacencies/transcriptomics_cms123_adj_matrix_p154_Lambda_np{not man}.csv', index_col=0)

    # compare similarity of all networks to each other
    proteomics_ALL_net = proteomics_ALL_net.values
    transcriptomics_ALL_net = transcriptomics_ALL_net.values
    proteomics_123_net = proteomics_123_net.values
    transcriptomics_123_net = transcriptomics_123_net.values

    print(f'Similarity of proteomics_ALL_net to transcriptomics_ALL_net: {evaluate_reconstruction(proteomics_ALL_net, transcriptomics_ALL_net)}')
    print(f'Similarity of proteomics_ALL_net to proteomics_123_net: {evaluate_reconstruction(proteomics_ALL_net, proteomics_123_net)}')
    print(f'Similarity of proteomics_ALL_net to transcriptomics_123_net: {evaluate_reconstruction(proteomics_ALL_net, transcriptomics_123_net)}')
    print(f'Similarity of transcriptomics_ALL_net to proteomics_123_net: {evaluate_reconstruction(transcriptomics_ALL_net, proteomics_123_net)}')
    print(f'Similarity of transcriptomics_ALL_net to transcriptomics_123_net: {evaluate_reconstruction(transcriptomics_ALL_net, transcriptomics_123_net)}')
    print(f'Similarity of proteomics_123_net to transcriptomics_123_net: {evaluate_reconstruction(proteomics_123_net, transcriptomics_123_net)}')




# sys.stdout.close()
# sys.stdout = original_stdout



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

