import numpy as np
import matplotlib.pyplot as plt

# Parameters for Gaussian bell curve (mean and standard deviation)
mu = 0
sigma = 2

# Generate data points
x = np.linspace(-5, 5, 400)
y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# Plotting the Gaussian bell curve
plt.figure(figsize=(8, 5))
# plot y against x with black line
plt.plot(x, y, '-', label='Posterior Distribution')
# plt.title('')
plt.xlabel(r'$z_k^{\lambda^j}$', fontsize=20)
plt.ylabel('Probability Density', fontsize=16)
# remove ylabels
plt.yticks([])
plt.xticks([])
# keep line only for x axis
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.legend(loc='upper left')
# place dot at x = 0 and y = 0.2
plt.plot(0, 0.2, 'ro')
plt.text(0.1, 0.2, r'$\mu^*_k$', fontsize=20, color='red')
plt.grid(False)
plt.show()
