import numpy as np
import scipy.stats as stats
from itertools import combinations

mat1 = np.array([[1,2,3],[4,5,6]])

sum = np.sum(mat1, axis=1)
print(sum)


edges = range(10)
zk_vec = np.random.rand(10, len(edges))
thetas = np.zeros((10, len(edges)))

post_distro = stats.norm(loc=0.5, scale=np.sqrt(0.5))

epsilon = 1e-5
for e_k in edges:
    thetas[:, e_k] = post_distro.cdf(zk_vec[:,e_k] + epsilon) - post_distro.cdf(zk_vec[:,e_k] - epsilon) 


print(thetas)

test_matrix = np.array([[1,2,3],
                        [0,0,0],
                        [4,4,4]])
test_matrix2 = np.array([[1,2,3],
                        [0,0,0],
                        [4,4,4]])
# stack the matrices vertically along a 3rd dimension
stacked_matrix = np.stack((test_matrix, test_matrix2), axis=2)
print(stacked_matrix.shape)

stacked_matrix_reshaped = stacked_matrix.reshape(stacked_matrix.shape[2], -1)
print(stacked_matrix_reshaped)



def estimate_lambda_wp(data, b, Q, p_k_matrix, zks, lambda_range, prior_matrix):
    n, p = data.shape
    results = []

    # ... rest of your code ...

    ######## POSTERIOR DISTRIBUTION ########
    mus = np.array(mus)
    vars = np.array(vars)
    psis = np.array(psis)

    # Vectorized computation of post_mu and post_var
    post_mu = (mus * tau_tr**2 + psis * vars) / (vars + tau_tr**2)
    post_var = (vars * tau_tr**2) / (vars + tau_tr**2)

    # Since the normal distribution parameters are now arrays, we can't use scipy.stats.norm
    # We'll compute the CDF values directly using the formula for the normal distribution CDF
    epsilon = 1e-5
    z_scores_plus = (z_mat + epsilon - post_mu[:, None]) / np.sqrt(post_var)[:, None]
    z_scores_minus = (z_mat - epsilon - post_mu[:, None]) / np.sqrt(post_var)[:, None]
    
    # Compute CDF values using the error function, which is related to the normal distribution CDF
    thetas = 0.5 * (scipy.special.erf(z_scores_plus / np.sqrt(2)) - scipy.special.erf(z_scores_minus / np.sqrt(2)))

    # ... rest of your code ...
