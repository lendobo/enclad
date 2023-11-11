import numpy as np
import matplotlib.pyplot as plt

# Load the determinant values from the file
file_path = 'determinant_values.txt'
det_values = np.loadtxt(file_path)
# check min and max values
print(np.min(det_values), np.max(det_values))

# Summarize the data
summary = {
    'Min Value': np.min(det_values),
    'Max Value': np.max(det_values),
    'Mean Value': np.mean(det_values),
    'Median Value': np.median(det_values),
    'Standard Deviation': np.std(det_values)
}

# Plot the determinant values over the optimization process
plt.figure(figsize=(10, 6))
plt.plot(det_values)
plt.title('Determinant Values of the Precision Matrix during Optimization')
plt.xlabel('Iteration')
plt.ylabel('Determinant Value')
plt.grid(True)
plt.yscale('log')  # Set y-axis to logarithmic scale for better visualization
plt.show(), summary
