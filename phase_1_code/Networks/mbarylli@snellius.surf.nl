import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from random import sample
import random
from numpy.random import multivariate_normal
from scipy.special import comb, erf
import scipy.stats as stats
from scipy.linalg import block_diag, eigh, inv
from sklearn.covariance import empirical_covariance, GraphicalLasso
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from mpi4py import MPI
from tqdm import tqdm
import sys
import pickle
import warnings
import os


# Activate the automatic conversion of numpy objects to R objects
numpy2ri.activate()

# Define the R function for weighted graphical lasso
ro.r('''
weighted_glasso <- function(data, penalty_matrix, nobs) {
  library(glasso)
  result <- glasso(s=as.matrix(data), rho=penalty_matrix, nobs=nobs)
  return(list(precision_matrix=result$wi, edge_counts=result$wi != 0))
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
    def __init__(self, data, prior_matrix, b, Q, rank, size):
        self.data = data
        self.prior_matrix = prior_matrix
        self.p = data.shape[1]
        self.n = data.shape[0]
        self.Q = Q
        self.subsample_indices = self.get_subsamples_indices(self.n, b, Q, rank, size)

    @staticmethod
    def generate_synth_data(p, n):
        if p == 100:
            m = random.choice([1,2,2])
        elif p == 300:
            m = random.choice([3,5,6])
        elif p == 500:
            m = random.choice([5,8,10])
        elif p == 1000:
            m = random.choice([10,15,20])
        else:
            m = 5

        # TRUE NETWORK
        G = nx.barabasi_albert_graph(p, m, seed=42)
        adj_matrix = nx.to_numpy_array(G)
        
        # PRECISION MATRIX
        precision_matrix = -0.5 * adj_matrix

        # Add to the diagonal to ensure positive definiteness
        # Set each diagonal entry to be larger than the sum of the absolute values of the off-diagonal elements
        # in the corresponding row
        diagonal_values = 2 * np.abs(precision_matrix).sum(axis=1)
        np.fill_diagonal(precision_matrix, diagonal_values)

        # Check if the precision matrix is positive definite
        # A simple check is to see if all eigenvalues are positive
        eigenvalues = np.linalg.eigh(precision_matrix)[0]
        is_positive_definite = np.all(eigenvalues > 0)

        # Compute the scaling factors for each variable (square root of the diagonal of the precision matrix)
        scaling_factors = np.sqrt(np.diag(precision_matrix))
        # Scale the precision matrix
        adjusted_precision = np.outer(1 / scaling_factors, 1 / scaling_factors) * precision_matrix

        covariance_mat = inv(adjusted_precision)

        # PRIOR MATRIX
        prior_matrix = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                if adj_matrix[i, j] != 0 and np.random.rand() < 0.95 :
                    prior_matrix[i, j] = 0.9
                    prior_matrix[j, i] = 0.9
                elif adj_matrix[i, j] == 0 and np.random.rand() < 0.05:
                    prior_matrix[i, j] = 0.9
                    prior_matrix[j, i] = 0.9
        np.fill_diagonal(prior_matrix, 0)

        prior_matrix = np.zeros((p, p))

        # DATA MATRIX
        np.random.seed(42)
        data = multivariate_normal(mean=np.zeros(G.number_of_nodes()), cov=covariance_mat, size=n)

        return data, prior_matrix, adj_matrix
    
    def get_subsamples_indices(self, n, b, Q, rank, size):
        """
        Generate a unique set of subsamples indices for a given MPI rank and size.
        """
        # Error handling: check if b and Q are valid 
        if b >= n:
            raise ValueError("b should be less than the number of samples n.")
        if Q > comb(n, b, exact=True):
            raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

        random.seed(42 + rank)  # Ensure each rank gets different subsamples
        subsamples_indices = set()

        # Each rank will attempt to generate Q/size unique subsamples
        subsamples_per_rank = Q // size
        attempts = 0
        max_attempts = subsamples_per_rank * 10  # to avoid an infinite loop

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
        S = empirical_covariance(sub_sample)

        # Number of observations
        nobs = sub_sample.shape[0]

        # Penalty matrix (adapt this to your actual penalty matrix logic)
        penalty_matrix = lambdax * np.ones((p,p)) # prior_matrix

        # print(f'P: {p}')

        # Call the R function from Python
        weighted_glasso = ro.globalenv['weighted_glasso']
        try:
            result = weighted_glasso(S, penalty_matrix, nobs)
            if 'error' in result.names:
                print(f"R Error or Warning: {result.rx('message')[0]}")
                return np.zeros((p,p)), np.zeros((p,p)), 0
            else:
                precision_matrix = np.array(result.rx('precision_matrix')[0])
                edge_counts = (np.abs(precision_matrix) > 1e-5).astype(int)
                return edge_counts, precision_matrix, 1
        except RRuntimeError as e:
            print(f"RRuntimeError: {e}")
            return np.zeros((p,p)), np.zeros((p,p)), 0


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



def main(rank, size):
    #######################
    p = 50             # number of variables (nodes)
    n = 500             # number of samples
    b = int(0.75 * n)   # size of sub-samples
    Q = 100             # number of sub-samples
    
    l_lo = 0.01
    l_hi = 0.4 
    lambda_range = np.linspace(l_lo, l_hi, 40)
    #######################

    data, prior_matrix, adj_matrix = QJSweeper.generate_synth_data(p, n)
    synthetic_QJ = QJSweeper(data, prior_matrix, b, Q, rank, size)

    # omics_QJ = 

    edge_counts_all, success_counts = synthetic_QJ.run_subsample_optimization(lambda_range)

    return edge_counts_all, p, n, Q

if __name__ == "__main__":
    # Initialize MPI communicator, rank, and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Check if running in SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        edge_counts, p, n, Q = main(rank=rank, size=size)

        # Gather the results at the root
        all_edges = comm.gather(edge_counts, root=0)

        if rank == 0:
            combined_edge_counts = np.zeros_like(all_edges[0])

            # Sum the edge counts across all ranks for each lambda value
            for edge_count in all_edges:
                combined_edge_counts += edge_count

            # Save combined results
            with open(f'net_results/combined_edge_counts_all.pkl', 'wb') as f:
                pickle.dump(combined_edge_counts, f)

            # Transfer results to $HOME
            os.system("cp -r net_results/ $HOME/phase_1_code/Networks/")

    else:
        # If no SLURM environment, run for entire lambda range
        edge_counts, p, n, Q = main(rank=1, size=1)

        # Save results to a pickle file
        with open(f'net_results/edge_counts_all_{p}_{n}_{Q}_fullrange.pkl', 'wb') as f:
            pickle.dump(edge_counts, f)
