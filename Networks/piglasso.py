import numpy as np
import pandas as pd
import networkx as nx
import math
import matplotlib.pyplot as plt
from random import sample
import random
from numpy.random import multivariate_normal
from scipy.special import comb, erf
import scipy.stats as stats
from scipy.linalg import block_diag, eigh, inv
from sklearn.covariance import empirical_covariance
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from mpi4py import MPI
from tqdm import tqdm
import sys
import pickle
import warnings
import os
import argparse
from scipy.stats import skewnorm


# Activate the automatic conversion of numpy objects to R objects
numpy2ri.activate()

# Define the R function for weighted graphical lasso
ro.r('''
weighted_glasso <- function(data, penalty_matrix, nobs) {
  library(glasso)
  tryCatch({
    result <- glasso(s=as.matrix(data), rho=penalty_matrix, nobs=nobs)
    return(list(precision_matrix=result$wi, edge_counts=result$wi != 0))
  }, error=function(e) {
    return(list(error_message=toString(e$message)))
  })
}
''')

class QJSweeper:
    """
    Class for parallel optimisation of the piGGM objective function, across Q sub-samples and J lambdas.

    Attributes
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.
    p : int
        The number of variables.

    Methods
    -------
    objective(precision_vector, S, lambda_np, lambda_wp, prior_matrix)
        The objective function for the piGGM optimization problem.

    optimize_for_q_and_j(params)
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        
    subsample_optimiser(b, Q, lambda_range)
        Optimizes the objective function for all sub-samples and lambda values, using optimize_for_q_and_j.
    """
    def __init__(self, data, prior_matrix, b, Q, rank=1, size=1):
        self.data = data
        self.prior_matrix = prior_matrix
        self.p = data.shape[1]
        self.n = data.shape[0]
        self.Q = Q
        self.subsample_indices = self.get_subsamples_indices(self.n, b, Q, rank, size, seed=args.seed)

    @staticmethod
    def generate_synth_data(p, n, fp_fn_chance, skew, density=0.03, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        density_params = {
        0.02: [(100, 1), (300, 3), (500, 5), (750, 8), (1000, 10)],
        0.03: [(100, 2), (300, 5), (500, 8), (750, 11), (1000, 15)],
        0.04: [(100, 2), (300, 6), (500, 10), (750, 15), (1000, 20)],
        0.1: [(100, 5), (300, 15), (500, 25), (750, 38), (1000, 50)],
        0.2: [(100, 10), (300, 30), (500, 50), (750, 75), (1000, 100)]
    }

        # Determine m based on p and the desired density
        m = 20  # Default value if p > 1000
        closest_distance = float('inf')
        for size_limit, m_value in density_params[density]:
            distance = abs(p - size_limit)
            if distance < closest_distance:
                closest_distance = distance
                m = m_value
        

        # TRUE NETWORK
        G = nx.barabasi_albert_graph(p, m, seed=seed)
        adj_matrix = nx.to_numpy_array(G)

        
        # PRECISION MATRIX
        precision_matrix = adj_matrix

        # Try adding a small constant to the diagonal until the matrix is positive definite
        small_constant = 0.01
        is_positive_definite = False
        while not is_positive_definite:
            np.fill_diagonal(precision_matrix, precision_matrix.diagonal() + small_constant)
            eigenvalues = np.linalg.eigh(precision_matrix)[0]
            is_positive_definite = np.all(eigenvalues > 0)
            small_constant += 0.01  # Increment the constant

        # Compute the scaling factors for each variable (square root of the diagonal of the precision matrix)
        scaling_factors = np.sqrt(np.diag(precision_matrix))
        # Scale the precision matrix
        adjusted_precision = np.outer(1 / scaling_factors, 1 / scaling_factors) * precision_matrix

        covariance_mat = inv(adjusted_precision)

        # PRIOR MATRIX
        prior_matrix = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                if adj_matrix[i, j] != 0 and np.random.rand() < (1 - fp_fn_chance):
                    prior_matrix[i, j] = 1
                    prior_matrix[j, i] = 1
                elif adj_matrix[i, j] == 0 and np.random.rand() < fp_fn_chance:
                    prior_matrix[i, j] = 1
                    prior_matrix[j, i] = 1
        np.fill_diagonal(prior_matrix, 0)


        # DATA MATRIX
        data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)

        if skew != 0:
            print('APPLYING SKEW: ', skew)
            # Determining which columns to skew
            columns_to_skew = np.random.choice(data.shape[1], size=int(0.2 * data.shape[1]), replace=False)
            left_skew_columns = columns_to_skew[:len(columns_to_skew) // 2]
            right_skew_columns = columns_to_skew[len(columns_to_skew) // 2:]

            # Applying skewness
            for col in left_skew_columns:
                data[:, col] += skewnorm.rvs(-skew, size=n)  # Left skew
            for col in right_skew_columns:
                data[:, col] += skewnorm.rvs(skew, size=n)  # Right skew

        return data, prior_matrix, adj_matrix
    
    def get_subsamples_indices(self, n, b, Q, rank, size, seed=42):
        """
        Generate a unique set of subsamples indices for a given MPI rank and size.
        """
        # Error handling: check if b and Q are valid 
        if b >= n:
            raise ValueError("b should be less than the number of samples n.")
        if Q > comb(n, b, exact=True):
            raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

        random.seed(seed + rank)  # Ensure each rank gets different subsamples
        subsamples_indices = set()

        # Each rank will attempt to generate Q/size unique subsamples
        subsamples_per_rank = Q // size
        attempts = 0
        max_attempts = 10e+5  # to avoid an infinite loop

        while len(subsamples_indices) < subsamples_per_rank and attempts < max_attempts:
            # Generate a random combination
            new_comb = tuple(sorted(sample(range(n), b)))
            subsamples_indices.add(new_comb)
            attempts += 1

        if attempts == max_attempts:
            raise Exception(f"Rank {rank}: Max attempts reached when generating subsamples.")

        return list(subsamples_indices)

    def optimize_for_q_and_j(self, single_subsamp_idx, lambdax):
        """
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        Parameters
        ----------
        subsamples_indices : array-like, shape (b)
            The indices of the sub-sample.
        lambdax : float
            The lambda value.

        Returns
        -------
        selected_sub_idx : array-like, shape (b)
            The indices of the sub-sample.
        lambdax : float
            The lambda value.
        edge_counts : array-like, shape (p, p)
            The edge counts of the optimized precision matrix.
        """
        data = self.data
        p = self.p
        prior_matrix = self.prior_matrix
        sub_sample = data[np.array(single_subsamp_idx), :]
        # try:
        S = empirical_covariance(sub_sample)
        # except Exception as e:
        #     print(f"Error in computing empirical covariance: {e}", file=sys.stderr)
        #     traceback.print_exc(file=sys.stderr)

        # Number of observations
        nobs = sub_sample.shape[0]

        # Penalty matrix (adapt this to your actual penalty matrix logic)
        penalty_matrix = lambdax * np.ones((p,p)) # prior_matrix

        # print(f'P: {p}')

        # Call the R function from Python
        weighted_glasso = ro.globalenv['weighted_glasso']
        # try:
        result = weighted_glasso(S, penalty_matrix, nobs)   
        # Check for an error message returned from R
        if 'error_message' in result.names:
            error_message = result.rx('error_message')[0][0]
            print(f"R Error: {error_message}", file=sys.stderr, flush=True)
            return np.zeros((p, p)), np.zeros((p, p)), 0
        else:
            precision_matrix = np.array(result.rx('precision_matrix')[0])
            edge_counts = (np.abs(precision_matrix) > 1e-5).astype(int)
            return edge_counts, precision_matrix, 1
        # except Exception as e:
        #     print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        #     return np.zeros((p, p)), np.zeros((p, p)), 0



    def run_subsample_optimization(self, lambda_range):
        """
        Run optimization on the subsamples for the entire lambda range.
        """
        edge_counts_all = np.zeros((self.p, self.p, len(lambda_range)))
        success_counts = np.zeros(len(lambda_range))

        # Replace this loop with calls to your actual optimization routine
        for q_idx in tqdm(self.subsample_indices):
            for lambdax in lambda_range:
                edge_counts, precision_matrix, success_check = self.optimize_for_q_and_j(q_idx, lambdax)
                l_idx = np.where(lambda_range == lambdax)[0][0]
                edge_counts_all[:, :, l_idx] += edge_counts
                success_counts[l_idx] += success_check

        return edge_counts_all, success_counts


def load_data(run_type, data_file, prior_file):
    data = pd.read_csv(data_file, index_col=0)
    # remove first column (CMS label)
    if prior_file:
        prior = pd.read_csv(prior_file, index_col=0)
    else:
        prior = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])))

    return data, prior

def main(rank, size, machine='local'):
    #######################
    p = args.p           # number of variables (nodes)
    n = args.n             # number of samples
    b = int(args.b_perc * n)   # size of sub-samples
    Q = args.Q             # number of sub-samples
    
    llo = args.llo
    lhi = args.lhi
    lamlen = args.lamlen
    lambda_range = np.linspace(llo, lhi, lamlen)
    #######################

    if args.run_type == 'synthetic':
        # Synthetic run
        data, prior_matrix, adj_matrix = QJSweeper.generate_synth_data(p, n, args.fp_fn, args.skew, args.dens, seed=args.seed)
        synthetic_QJ = QJSweeper(data, prior_matrix, b, Q, rank, size)

        edge_counts_all, success_counts = synthetic_QJ.run_subsample_optimization(lambda_range)

    elif args.run_type == 'proteomics' or args.run_type == 'transcriptomics':
        # Loading data
        cms_data = pd.read_csv(args.data_file, index_col=0)
        p = cms_data.shape[1]
        cms_array = cms_data.values

        # Checking for prior
        if args.prior_file:
            cms_omics_prior = pd.read_csv(args.prior_file, index_col=0)
        else:
            print('----------------\nNo prior supplied, defaulting to data-only run\n----------------')
            cms_omics_prior = pd.DataFrame(np.zeros((p,p)))

        prior_matrix = cms_omics_prior.values

        n = cms_array.shape[0]
        b = int(args.b_perc * n)

        print(f'Variables, Samples: {p, n}')
        print(cms_array[1])

        # scale and center 
        cms_array = (cms_array - cms_array.mean(axis=0)) / cms_array.std(axis=0)
        # run QJ Sweeper
        omics_QJ = QJSweeper(cms_array, prior_matrix, b, Q, rank, size)

        edge_counts_all, success_counts = omics_QJ.run_subsample_optimization(lambda_range)

    return edge_counts_all, p, n, Q

if __name__ == "__main__":
    # Initialize MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run QJ Sweeper with command-line arguments.')
    parser.add_argument('--p', type=int, default=50, help='Number of variables (nodes)')
    parser.add_argument('--n', type=int, default=500, help='Number of samples')
    parser.add_argument('--Q', type=int, default=800, help='Number of sub-samples')
    parser.add_argument('--b_perc', type=float, default=0.8, help='Size of sub-samples (as a percentage of n)')
    parser.add_argument('--llo', type=float, default=0.01, help='Lower bound for lambda range')
    parser.add_argument('--lhi', type=float, default=0.4, help='Upper bound for lambda range')
    parser.add_argument('--lamlen', type=int, default=40, help='Number of points in lambda range')
    parser.add_argument('--run_type', type=str, default='synthetic', choices=['synthetic', 'proteomics', 'transcriptomics'], help='Type of run to execute')
    parser.add_argument('--data_file', type=str, default=None, help='omics data file (Protein / RNA))')
    parser.add_argument('--prior_file', type=str, default=None, help='adjacency matrix for prior')
    parser.add_argument('--cms', type=str, default='cmsALL', choices=['cmsALL', 'cms123'], help='CMS type to run for omics run')
    parser.add_argument('--fp_fn', type=float, default=0, help='Chance of getting a false negative or a false positive')
    parser.add_argument('--skew', type=float, default=0, help='Skewness of the data')
    parser.add_argument('--dens', type=float, default=0.03, help='Density of the synthetic network')
    parser.add_argument('--seed', type=int, default=42, help='Seed for generating synthetic data')

    args = parser.parse_args()

    # Check if running in SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        edge_counts, p, n, Q = main(rank=rank, size=size, machine='hpc')

        num_elements = p * p * args.lamlen
        sendcounts = np.array([num_elements] * size)
        displacements = np.arange(size) * num_elements

        if rank == 0:
            # Gather the results at the root
            all_edges = np.empty(size * num_elements, dtype=edge_counts.dtype)
        else:
            all_edges = None

        comm.Gatherv(sendbuf=edge_counts.flatten(), recvbuf=(all_edges, sendcounts, displacements, MPI.DOUBLE), root=0)

        if rank == 0:
            # Reshape all_edges back to the original shape (size, p, p, len(lambda_range))
            reshaped_edges = all_edges.reshape(size, p, p, args.lamlen)

            combined_edge_counts = np.sum(reshaped_edges, axis=0)


            # Save combined results
            with open(f'net_results/{args.run_type}_{args.cms}_edge_counts_all_pnQ{args.p}_{args.n}_{args.Q}_{args.llo}_{args.lhi}_ll{args.lamlen}_b{args.b_perc}_fpfn{args.fp_fn}_skew{args.skew}_dens{args.dens}.pkl', 'wb') as f:
                pickle.dump(combined_edge_counts, f)

            # Transfer results to $HOME
            os.system("cp -r net_results/ $HOME/thesis_code/Networks/")

    else:
        # If no SLURM environment, run for entire lambda range
        edge_counts, p, n, Q = main(rank=1, size=1, machine='local')
        print(edge_counts.dtype)

        # Save results to a pickle file
        with open(f'Networks/net_results/local_{args.run_type}_{args.cms}_edge_counts_all_pnQ{p}_{args.n}_{args.Q}_{args.llo}_{args.lhi}_ll{args.lamlen}_b{args.b_perc}_fpfn{args.fp_fn}_dens{args.dens}.pkl', 'wb') as f:
            pickle.dump(edge_counts)


# scp mbarylli@snellius.surf.nl:"phase_1_code/Networks/net_results/omics_edge_counts_all_pnQ\(100\,\ 106\,\ 300\).pkl" net_results/
 