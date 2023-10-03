import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import inv, eigh


p = 5
testmat = np.zeros((p, p))
for i in range(p):
    for j in range(i, p):
        testmat[i, j] = 1
        testmat[j, i] = 1
