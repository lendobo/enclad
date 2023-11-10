import numpy as np
from scipy.optimize import curve_fit
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


import numpy as np
from scipy.optimize import curve_fit

# Define a linear function for curve fitting
def linear_func(x, a, b):
    return a * x + b

def fit_lines_and_get_error(index, lambdas, edge_counts, left_bound, right_bound):
    # Only consider data points within the specified bounds
    left_data = lambdas[left_bound:index+1]
    right_data = lambdas[index:right_bound]

    if len(left_data) < 3 or len(right_data) < 3:
        return np.inf

    # Fit lines to the left and right of current index within bounds
    # print(index)
    params_left, _ = curve_fit(linear_func, left_data, edge_counts[left_bound:index+1])
    # print(index)
    params_right, _ = curve_fit(linear_func, right_data, edge_counts[index:right_bound])
    
    # Calculate fit errors within bounds
    error_left = np.sum((linear_func(left_data, *params_left) - edge_counts[left_bound:index+1]) ** 2)
    error_right = np.sum((linear_func(right_data, *params_right) - edge_counts[index:right_bound]) ** 2)
    
    return error_left + error_right

def find_knee_point(lambda_range, edge_counts_all, left_bound, right_bound):
    errors = [fit_lines_and_get_error(i, lambda_range, edge_counts_all, left_bound, right_bound) 
              for i in range(left_bound, right_bound)]
    knee_point_index = np.argmin(errors) + left_bound
    return knee_point_index

def find_all_knee_points(lambda_range, edge_counts_all):
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    # Find the main knee point across the full range
    main_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, len(lambda_range))
    main_knee_point = lambda_range[main_knee_point_index]
    
    # For the left knee point, consider points to the left of the main knee point
    left_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, main_knee_point_index)
    left_knee_point = lambda_range[left_knee_point_index]
    
    # For the right knee point, consider points to the right of the main knee point
    # Update the bounds to ensure the fit_lines_and_get_error function considers only the right subset
    right_knee_point_index = find_knee_point(lambda_range, edge_counts_all, main_knee_point_index, len(lambda_range))
    right_knee_point = lambda_range[right_knee_point_index]
    
    return left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, main_knee_point_index, right_knee_point_index

# Main code
if __name__ == "__main__":
    #### Main code ####
    p = 100             # number of variables (nodes)
    n = 500             # number of samples
    b = int(0.75 * n)   # size of sub-samples
    Q = 800             # number of sub-samples

    lowerbound = 0.01
    upperbound = 0.4
    lambda_range = np.linspace(lowerbound, upperbound, 60)

    filename_edges = f'net_results/synthetic_edge_counts_all_pnQ{p}_{n}_{Q}_{lowerbound}_{upperbound}_60.pkl'
    with open(filename_edges, 'rb') as f:
        edge_counts_all = pickle.load(f)

    # divide each value in edge_counts_all by 2*Q
    edge_counts_all = edge_counts_all / (2 * Q)


    left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, knee_point_index, right_knee_point_index = find_all_knee_points(lambda_range, edge_counts_all)
    print("Left Knee Point at lambda =", left_knee_point)
    print("Main Knee Point at lambda =", main_knee_point)
    print("Right Knee Point at lambda =", right_knee_point)

    # We will now plot the additional lines: the right red line and the left magenta line
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    plt.figure(figsize=(14, 7))
    plt.plot(lambda_range, edge_counts_all, 'bo', label='Edge Counts', alpha = 0.5)

    # Fit and plot the lines for the left knee point
    left_data = lambda_range[:left_knee_point_index+1]
    left_fit_params, _ = curve_fit(linear_func, left_data, edge_counts_all[:left_knee_point_index+1])
    plt.plot(left_data, linear_func(left_data, *left_fit_params), 'r-', label='Left Fit')

    # Fit and plot the line between the left knee point and the main knee point (right red line)
    left_knee_to_main_data = lambda_range[left_knee_point_index:knee_point_index+1]
    left_knee_to_main_fit_params, _ = curve_fit(linear_func, left_knee_to_main_data, edge_counts_all[left_knee_point_index:knee_point_index+1])
    plt.plot(left_knee_to_main_data, linear_func(left_knee_to_main_data, *left_knee_to_main_fit_params), 'r--', label='Right of Left Knee Fit')

    # Fit and plot the lines for the main knee point
    main_left_data = lambda_range[:knee_point_index]
    main_right_data = lambda_range[knee_point_index:]
    main_left_fit_params, _ = curve_fit(linear_func, main_left_data, edge_counts_all[:knee_point_index])
    main_right_fit_params, _ = curve_fit(linear_func, main_right_data, edge_counts_all[knee_point_index:])
    plt.plot(main_left_data, linear_func(main_left_data, *main_left_fit_params), 'g-', label='Main Left Fit')
    plt.plot(main_right_data, linear_func(main_right_data, *main_right_fit_params), 'g-', label='Main Right Fit')

    # Fit and plot the line between the main knee point and the right knee point (left magenta line)
    main_to_right_knee_data = lambda_range[knee_point_index:right_knee_point_index+1]
    main_to_right_knee_fit_params, _ = curve_fit(linear_func, main_to_right_knee_data, edge_counts_all[knee_point_index:right_knee_point_index+1])
    plt.plot(main_to_right_knee_data, linear_func(main_to_right_knee_data, *main_to_right_knee_fit_params), 'm--', label='Left of Right Knee Fit')

    # Fit and plot the lines for the right knee point
    right_data = lambda_range[right_knee_point_index:]
    right_fit_params, _ = curve_fit(linear_func, right_data, edge_counts_all[right_knee_point_index:])
    plt.plot(right_data, linear_func(right_data, *right_fit_params), 'm-', label='Right Fit')

    # Mark the knee points on the plot
    plt.axvline(x=left_knee_point, color='r', linestyle='--', label='Left Knee Point')
    plt.axvline(x=main_knee_point, color='g', linestyle='--', label='Main Knee Point')
    plt.axvline(x=right_knee_point, color='m', linestyle='--', label='Right Knee Point')

    plt.xlabel('Lambda')
    plt.ylabel('Edge Counts')
    plt.title('Knee Points and Fitted Lines')
    plt.legend()
    plt.grid()
    plt.show()
