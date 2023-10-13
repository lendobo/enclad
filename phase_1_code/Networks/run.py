import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import inv, eigh
from numpy.random import multivariate_normal
from itertools import combinations

p = 10

num_priors = 3

checks = [ind / (num_priors + 2) for ind in range(num_priors)]

booleans = [np.random.rand() < ind / num_priors for ind in range(num_priors)]

print(checks)
