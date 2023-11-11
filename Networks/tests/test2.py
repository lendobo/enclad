import numpy as np

results = [[0,0.2,[[1,2,3],[4,5,6]]], [1, 0.3, [[1,2,3],[4,5,6]]]]
lambda_range = np.array([0.2, 0.3])

for q, lambdax, edge_counts in results:
    l = np.where(lambda_range == lambdax)[0][0]
    print(l)