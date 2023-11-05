import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import numpy as np


# Activate the automatic conversion of numpy objects to R objects
numpy2ri.activate()

# Define the R function for weighted graphical lasso
ro.r('''
weighted_glasso <- function(data, penalty_matrix, nobs) {
  library(glasso)
  result <- glasso(s=as.matrix(data), rho=penalty_matrix, nobs=nobs)
  return(list(precision_matrix=result$wi, edge_counts=result$wi != 0))
}
''')

# Create a numpy array as an example input
sub_sample = np.random.rand(5, 5)  # Replace with your actual data
penalty_matrix = np.random.rand(5, 5)  # Define your penalty matrix
nobs = sub_sample.shape[0]  # Number of observations

# Call the R function from Python
weighted_glasso = ro.globalenv['weighted_glasso']
result = weighted_glasso(sub_sample, penalty_matrix, nobs)

# Extract the precision matrix and edge counts
precision_matrix = np.array(result[0])
edge_counts = np.array(result[1], dtype=int)

print(precision_matrix)
print(edge_counts)