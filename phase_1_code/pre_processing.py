# %%
import scanpy as sc

# %%

# Load the .h5ad file
adata = sc.read("data/CRC_UNB_10X_E-MTAB-8107.h5ad")

# Preprocessing
# Normalize the data
sc.pp.normalize_total(adata, target_sum=1e4)

# # Logarithmize the data
# sc.pp.log1p(adata)

# # Identify highly-variable genes
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# # Keep only highly variable genes
# adata = adata[:, adata.var.highly_variable]

# # Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
# sc.pp.regress_out(adata, ["n_counts", "percent_mito"])

# # Scale each gene to unit variance, clip values exceeding standard deviation 10.
# sc.pp.scale(adata, max_value=10)

# Dimensionality reduction
# Run PCA
sc.tl.pca(adata, svd_solver="arpack")

# Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Clustering the graph
sc.tl.leiden(adata)

# Run UMAP
sc.tl.umap(adata)

# visualize the results
sc.pl.umap(adata, color=["leiden", "cell_type"], save="_preprocessing.png")

# Save the cluster labels and other metadata into a .tsv file
metadata = adata.obs
metadata.to_csv("metadata.tsv", sep="\t")

# %%
