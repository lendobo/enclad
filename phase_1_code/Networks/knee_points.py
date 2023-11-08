import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Define a linear function for curve fitting
def linear_func(x, a, b):
    return a * x + b

def fit_lines_and_get_error(index, lambdas, edge_counts):
    # Extracted the edge count conversion to outside of this function
    left_data = lambdas[:index+1]
    right_data = lambdas[index:]

    if len(left_data) < 2 or len(right_data) < 2:
        return np.inf

    # Fit lines to the left and right of current index
    params_left, _ = curve_fit(linear_func, left_data, edge_counts[:index+1])
    params_right, _ = curve_fit(linear_func, right_data, edge_counts[index:])
    
    # Calculate fit errors
    error_left = np.sum((linear_func(left_data, *params_left) - edge_counts[:index+1]) ** 2)
    error_right = np.sum((linear_func(right_data, *params_right) - edge_counts[index:]) ** 2)
    
    return error_left + error_right


def find_single_knee_point(lambdas, edge_counts_all):
    # Calculate the total fit error for each lambda and find the 'knee-point'
    edge_counts = [np.sum(matrix)/2 for matrix in np.rollaxis(edge_counts_all, -1)]
    errors = [fit_lines_and_get_error(i, lambdas, edge_counts) for i in range(len(lambdas))]
    knee_point_index = np.argmin(errors)
    return knee_point_index

#### Main code ####

lambda_range = np.linspace(0.01, 0.4, 40)

p_range = 100
n = 300
b_values = 250 # int((2 / 3) * n)
Q_values = 300 # int((1 / 3) * n)

filename_edges = 'net_results/synthetic_edge_counts_all_pnQ250_800_1000_0.01_0.4_40.pkl'
with open(filename_edges, 'rb') as f:
    edge_counts_all = pickle.load(f)

# divide each value in edge_counts_all by 2*Q
edge_counts_all = edge_counts_all / (2 * Q_values)


print("[Nodes, Nodes, Lambdas]:", edge_counts_all.shape)

knee_point_index = find_single_knee_point(lambda_range, edge_counts_all)
knee_point = lambda_range[knee_point_index]
print("Found main knee-point at lambda =", knee_point)

left_knee_point_index = find_single_knee_point(lambda_range[:knee_point_index+1], edge_counts_all[..., :knee_point_index+1])
right_knee_point_index = knee_point_index + find_single_knee_point(lambda_range[knee_point_index:], edge_counts_all[..., knee_point_index:])

print("Found left knee-point at index:", left_knee_point_index)
print("Found right knee-point at index:", right_knee_point_index)
print("Left knee-point at lambda =", lambda_range[left_knee_point_index])
print("Right knee-point at lambda =", lambda_range[right_knee_point_index])

# Convert the 3D array to edge counts for each lambda for plotting
edge_counts = [np.sum(matrix)/2 for matrix in np.rollaxis(edge_counts_all, -1)]

# Fit curves again around optimal knee-point for plotting
left_data = lambda_range[:knee_point_index]
right_data = lambda_range[knee_point_index+1:]
params_left, _ = curve_fit(linear_func, left_data, edge_counts[:knee_point_index])
params_right, _ = curve_fit(linear_func, right_data, edge_counts[knee_point_index+1:])

# Fit curves around left knee-point for plotting
left_data_l = lambda_range[:left_knee_point_index+1]
right_data_l = lambda_range[left_knee_point_index+1:]
params_left_l, _ = curve_fit(linear_func, left_data_l, edge_counts[:left_knee_point_index+1])
params_right_l, _ = curve_fit(linear_func, right_data_l, edge_counts[left_knee_point_index+1:])

# Fit curves around right knee-point for plotting
left_data_r = lambda_range[:right_knee_point_index+1]
right_data_r = lambda_range[right_knee_point_index+1:]
params_left_r, _ = curve_fit(linear_func, left_data_r, edge_counts[:right_knee_point_index+1])
params_right_r, _ = curve_fit(linear_func, right_data_r, edge_counts[right_knee_point_index+1:])



# plot the fitted curves to the left and right of the knee point
plt.figure(figsize=(10, 6))
plt.scatter(lambda_range, edge_counts, label="Data")
plt.plot(lambda_range[:knee_point_index], linear_func(lambda_range[:knee_point_index], *params_left), color="green", label="Left Fit", alpha=0.4)
plt.plot(lambda_range[knee_point_index+1:], linear_func(lambda_range[knee_point_index+1:], *params_right), color="green", label="Right Fit", alpha=0.4)
plt.plot(lambda_range[:left_knee_point_index+1], linear_func(lambda_range[:left_knee_point_index+1], *params_left_l), color="orange", label="Left, L", alpha=0.4)
plt.plot(lambda_range[left_knee_point_index+1:], linear_func(lambda_range[left_knee_point_index+1:], *params_right_l), color="orange", label="right, L", alpha=0.4)
plt.plot(lambda_range[:right_knee_point_index+1], linear_func(lambda_range[:right_knee_point_index+1], *params_left_r), color="purple", label="Left, R", alpha=0.4)
plt.plot(lambda_range[right_knee_point_index+1:], linear_func(lambda_range[right_knee_point_index+1:], *params_right_r), color="purple", label="right, R", alpha=0.4)
plt.axvline(x=lambda_range[knee_point_index], color="green", linestyle="--", label="Knee Point")
plt.axvline(x=lambda_range[left_knee_point_index], color="orange", linestyle="--", label="Left Knee Point")
plt.axvline(x=lambda_range[right_knee_point_index], color="purple", linestyle="--", label="Right Knee Point")
#draw horizontal line at 0
plt.axhline(y=0, color="black", alpha=0.5)
plt.axvline(x=0, color="black", alpha=0.5)
plt.xlabel("Penalty value")
plt.ylabel("Number of edges")
plt.legend()
plt.show()