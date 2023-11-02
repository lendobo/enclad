import numpy as np
import matplotlib.pyplot as plt
import pickle

# Define lambda values for the filenames
lams = [
    (0.0, 0.011428571428571429),
    (0.022857142857142857, 0.03428571428571429),
    (0.045714285714285714, 0.05714285714285714),
    (0.06857142857142857, 0.08)
]

# Load edge counts from files
edge_counts = []
for i in range(4):
    filename = f'net_results/edge_counts_all(300, 500, 800)_lams({lams[i][0]}, {lams[i][1]})_n{i}.pkl'
    with open(filename, 'rb') as f:
        data = np.array(pickle.load(f))
        summed_data = np.sum(data, axis=(0, 1)) // 2
        edge_counts.append(summed_data)

for i in range(1,5):
    filename = f'net_results/edge_counts_all(300, 500, 800)_n{i}.pkl'
    with open(filename, 'rb') as f:
        data = np.array(pickle.load(f))
        summed_data = np.sum(data, axis=(0, 1)) // 2
        edge_counts.append(summed_data)


# Concatenate edge_counts into a 1D array
edge_counts_concat = np.concatenate(edge_counts)

lambdas = np.linspace(0, 0.4, len(edge_counts_concat))

# l_lo = 0 # 0.04050632911392405
# l_hi = 0.4 # 0.1569620253164557
# lambda_range = np.linspace(l_lo, l_hi, 40)

# plot concatenated edge counts agains lambda_range
plt.figure(figsize=(10, 6))
plt.scatter(lambdas, edge_counts_concat, label="Data")
plt.xlabel('lambda')
plt.ylabel('edge count')
plt.title('Edge count vs. lambda')
plt.show()
