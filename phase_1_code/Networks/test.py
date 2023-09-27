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