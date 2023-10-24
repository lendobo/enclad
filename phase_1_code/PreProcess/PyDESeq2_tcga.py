# %%
import pandas as pd

# Read the metadata file
metadata = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/mergedPhenotype/cms_labels_public_tcga.txt', sep='\t', header=None)
# Correcting the column names for metadata
metadata.columns = ['SampleID', 'c1', 'c2', 'c3', 'CancerSubtype']

# Read the count matrix
count_matrix = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/TCGACRC_expression.tsv', sep='\t', index_col=0)

# %%
# Display the first few rows of both dataframes for inspection
metadata.head()

# %%
count_matrix.head()

# %%
# Get the sample IDs from both datasets
metadata_sample_ids = set(metadata['SampleID'])
count_matrix_sample_ids = set(count_matrix.columns)

# Find the common sample IDs
common_sample_ids = list(metadata_sample_ids.intersection(count_matrix_sample_ids))

# Find the sample IDs present in metadata but not in count matrix and vice versa
metadata_not_in_count_matrix = metadata_sample_ids.difference(count_matrix_sample_ids)
count_matrix_not_in_metadata = count_matrix_sample_ids.difference(metadata_sample_ids)

len(common_sample_ids), len(metadata_not_in_count_matrix), len(count_matrix_not_in_metadata)


# %%
# Filter and reorder the metadata and count matrix based on the common sample IDs
filtered_metadata = metadata[metadata['SampleID'].isin(common_sample_ids)]
filtered_count_matrix = count_matrix[common_sample_ids]

# Ensure the order of samples in the count matrix matches the order in the metadata
filtered_metadata = filtered_metadata.set_index('SampleID').loc[filtered_count_matrix.columns].reset_index()

filtered_count_matrix.head()

# %%
# Correcting the column name and filtering the count matrix
sample_ids_for_analysis = filtered_metadata['index'].values
filtered_count_matrix = filtered_count_matrix[sample_ids_for_analysis]

# filtered_metadata.shape, filtered_count_matrix.shape
filtered_metadata.head()

# %%
# Concatenate the metadata with the count matrix
deseq2_table = pd.concat([filtered_metadata[['SampleID', 'CancerSubtype']], filtered_count_matrix.T], axis=1)


# %%
