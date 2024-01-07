# %%
import pandas as pd

pathway_df = pd.read_csv('data/Pathway_Enrichment_Info.csv')

pathway_df = pathway_df.drop_duplicates(subset='description', keep='first')

num_duplicates = pathway_df['description'].duplicated().sum()
print("Number of duplicate descriptions:", num_duplicates)

# Filter the dataframe to include only pathways with '# genes' between 1 and 25
filtered_pathway_df = pathway_df[(pathway_df['# genes'] >= 10) & (pathway_df['# genes'] <= 25)]
# only keep first X
filtered_pathway_df = filtered_pathway_df.head(80)

# Extract the descriptions of these pathways
filtered_pathways = filtered_pathway_df['description'].tolist()

# Display the number of pathways and first few for inspection
num_filtered_pathways = len(filtered_pathways)
filtered_pathways[:10], num_filtered_pathways  # Displaying first 10 for brevity



# %%
import numpy as np

matrixA = np.array([[1, 2, 3], [4, 5, 6]])
matrixB = np.array([[1, 2, 3], [4, 5, 6]])

diff_kernel_knock_aggro = [matrixA + k for k in range(10)]
diff_kernel_knock_non_mesench = [matrixB + k for k in range(10)]


# # Calculate diffusion kernels and GDD
# diff_kernel_knock_aggro = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_aggro, t) for t in t_values]
# diff_kernel_knock_non_mesench = [laplacian_exponential_kernel_eigendecomp(knockdown_laplacian_non_mesench, t) for t in t_values]

gdd_values_trans = np.linalg.norm(np.array(diff_kernel_knock_non_mesench) - np.array(diff_kernel_knock_aggro), axis=(1, 2), ord='fro')

print(gdd_values_trans.shape)

# %%
import pandas as pd

N = 670000000

prob_not = [(N - k)/ N for k in range(1, 9999)]

prob_not = np.array(prob_not)

# turn it into a pandas series
prob_not = pd.Series(prob_not).prod()

prob_yes = 1 - prob_not

print(prob_yes)
