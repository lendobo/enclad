# %%
import pandas as pd
import sklearn.covariance as covariance
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# %%
# Load the entire data file
file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/Linkedomics1/Human__CPTAC_COAD__UNC__RNAseq__HiSeq_RNA__03_01_2017__BCM__Gene__BCM_RSEM_UpperQuartile_log2.cct"
df = pd.read_csv(file_path, sep="\t")
df = df.set_index("attrib_name")


# subtract row-mean from each value
df = df.sub(df.mean(axis=1), axis=0)
# divide each value by row-std
df = df.div(df.std(axis=1), axis=0)

df

# %%
### SUBSAMPLING ###
# Randomly select X genes
selected_genes = df.sample(n=500, axis=0, random_state=42)
# Randomly select Y samples for selected_genes
df_sub = selected_genes.sample(n=100, axis=1, random_state=42)

# %%
### DATA INSPECTION ###

# MAKE A NUMPY ARRAY FROM DF
arr = df.to_numpy()
np.isfinite(arr).all()

# # Compute the condition number of the covariance matrix
# cov_matrix = np.cov(df, rowvar=False)
# condition_number = np.linalg.cond(cov_matrix)
# print("Condition number:", condition_number)

# %%
# Calling Glasso algorithm
edge_model = covariance.GraphicalLassoCV(cv=10)
# df /= df.std(axis=0)
edge_model.fit(df)
# the precision(inverse covariance) matrix that we want
p = edge_model.precision_

# %%
# remove diagonal
np.fill_diagonal(p, 0)

# %%
# Step 3: Create Graph from Precision Matrix
G = nx.Graph(abs(p) > 0)  # Threshold to exclude small values

# Draw the Graph
plt.figure(figsize=(10, 10))
nx.draw(G, with_labels=False, node_size=20)
plt.title("Graphical Model Representing Conditional Dependencies")
plt.show()

# %%
# Print nodes with highest degree, with gene names
degrees = G.degree()
sorted(degrees, key=lambda x: x[1], reverse=True)[:10]

# index df at rows 21, 80, 19, 60 and 9
df.iloc[[21, 80, 19, 60, 9]]

# %%
# Export G to GEphi
nx.write_gexf(G, "glasso_graph.gexf")
