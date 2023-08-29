# %%
from scipy.stats import lognorm
import numpy as np
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

# Set the seed for reproducibility
np.random.seed(42)

# Number of genes and samples
num_genes = 500
num_samples = 50

# Parameters for the log-normal distribution
mean_log = 0
std_dev_log = 1

# Generate synthetic RNA-Seq counts
synthetic_data = lognorm.rvs(
    s=std_dev_log, scale=np.exp(mean_log), size=(num_genes, num_samples)
)

# Compute the condition number of the covariance matrix
cov_matrix = np.cov(synthetic_data, rowvar=False)
condition_number = np.linalg.cond(cov_matrix)
print("Condition number:", condition_number)

# Standardizing the synthetic data
scaler = StandardScaler()
synthetic_data_scaled = scaler.fit_transform(synthetic_data)

# Displaying the first few rows
synthetic_data_scaled[:5, :5]

# %%
### GRAPHICAL LASSO ###

# Calling Graphical Lasso algorithm
edge_model_synthetic = GraphicalLassoCV(cv=10)

# Fitting the model to the standardized synthetic data
edge_model_synthetic.fit(synthetic_data_scaled)

# Retrieving the precision (inverse covariance) matrix
precision_matrix_synthetic = edge_model_synthetic.precision_

# Displaying the top-left 5x5 submatrix of the precision matrix
# precision_matrix_synthetic[:5, :5]

# %%
