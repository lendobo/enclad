import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('net_results/edge_counts_all(300, 500, 800)_lams(0.0, 0.011428571428571429)_n0.pkl', 'rb') as f:
    edge_counts_0 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_lams(0.022857142857142857, 0.03428571428571429)_n1.pkl', 'rb') as f:
    edge_counts_1 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_lams(0.045714285714285714, 0.05714285714285714)_n2.pkl', 'rb') as f:
    edge_counts_2 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_lams(0.06857142857142857, 0.08)_n3.pkl', 'rb') as f:
    edge_counts_3 = np.array(pickle.load(f))

with open('net_results/edge_counts_all(300, 500, 800)_n0.pkl', 'rb') as f:
    edge_counts_0 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_n1.pkl', 'rb') as f:
    edge_counts_1 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_n2.pkl', 'rb') as f:
    edge_counts_2 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_n3.pkl', 'rb') as f:
    edge_counts_3 = np.array(pickle.load(f))
with open('net_results/edge_counts_all(300, 500, 800)_n4.pkl', 'rb') as f:
    edge_counts_4 = np.array(pickle.load(f))


# sum edge_counts_1 over the first 2 axes
edge_counts_0 = np.sum(edge_counts_0, axis=(0, 1)) // 2
x_0 = np.arange(0, len(edge_counts_0))
# sum edge_counts_1 over the first 2 axes
edge_counts_1 = np.sum(edge_counts_1, axis=(0, 1)) // 2
x_1 = np.arange(0, len(edge_counts_1))
# sum edge_counts_2 over the first 2 axes
edge_counts_2 = np.sum(edge_counts_2, axis=(0, 1)) // 2
x_2 = np.arange(0, len(edge_counts_2))
# sum edge_counts_3 over the first 2 axes
edge_counts_3 = np.sum(edge_counts_3, axis=(0, 1)) // 2
x_3 = np.arange(len(edge_counts_2), len(edge_counts_2) + len(edge_counts_3))
# sum edge_counts_4 over the first 2 axes
edge_counts_4 = np.sum(edge_counts_4, axis=(0, 1)) // 2
x_4 = np.arange(len(edge_counts_2) + len(edge_counts_3), len(edge_counts_2) + len(edge_counts_3) + len(edge_counts_4))

# # plot each edge coutn into the same plot
# plt.plot(x_2, edge_counts_2, label='n = 2')
# plt.plot(x_3, edge_counts_3, label='n = 3')
# plt.plot(x_4, edge_counts_4, label='n = 4')
# plt.legend()

# concatenate the edge counts
edge_counts = np.concatenate((edge_counts_0,edge_counts_1,edge_counts_2, edge_counts_3, edge_counts_4))
print(edge_counts.shape)

l_lo = 0 # 0.04050632911392405
l_hi = 0.4 # 0.1569620253164557
lambda_range = np.linspace(l_lo, l_hi, 40)[:32]

# plot concatenated edge counts agains lambda_range
plt.figure(figsize=(10, 6))
plt.scatter(lambda_range, edge_counts, label="Data")
plt.xlabel('lambda')
plt.ylabel('edge count')
plt.title('Edge count vs. lambda')
plt.show()
