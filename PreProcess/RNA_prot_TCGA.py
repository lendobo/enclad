# %%
import pandas as pd
import numpy as np


proteomic_file_path = "../data/Synapse/TCGA/TCGACRC_proteomics.csv"
rna_file_path = "../data/Synapse/TCGA/TCGACRC_expression.tsv"

# Reading the proteomic data (CSV format) and setting "GeneSymbol" as the index
proteomic_data = pd.read_csv(proteomic_file_path)
proteomic_data.set_index("GeneSymbol", inplace=True)

# write index to .txt file
with open("../data/Synapse/TCGA/Prot_List.csv", "w") as f:
    for s in proteomic_data.index:
        f.write(str(s) + "\n")


# Reading the RNA data (TSV format) and setting "feature" as the index
rna_data = pd.read_csv(rna_file_path, sep="\t")
rna_data.set_index("feature", inplace=True)

proteomic_data.head()

rna_data.head()

# find matching columns in both dataframes
matching_columns = list(set(proteomic_data.columns).intersection(rna_data.columns))

# filter the dataframes to only include the matching columns
proteomic_data_match = proteomic_data[matching_columns]
rna_data_match = rna_data[matching_columns]

# %%
rna_data.shape

# %%
proteomic_data.shape

# %%
# Transform protein indeces to RNA index format
trunc_prot_indices = [index.split('-')[0:3] for index in proteomic_data.columns]
trunc_prot_indices = ['-'.join(index) for index in trunc_prot_indices]


# Normalize the indices for the RNA data for consistency
trunc_rna_indices = [index.split('-')[0:3] for index in rna_data.columns]
trunc_rna_indices = ['-'.join(index) for index in trunc_rna_indices]

# Create sets for easier comparison
set_trunc_prot_indices = set(trunc_prot_indices)
set_trunc_rna_indices = set(trunc_rna_indices)

print(len(set_trunc_prot_indices))

# Find the intersection of the two sets to get the matching IDs
matching_ids = set_trunc_prot_indices.intersection(set_trunc_rna_indices)

# Count the number of matching IDs
num_matching_ids = len(matching_ids)

# Show some example matching IDs and the total number
list(matching_ids)[:10], num_matching_ids


# %%
# label file path
label_file_path = "../data/Synapse/mergedPhenotype/cms_labels_public_all.txt"

# Reading and filtering the label data
label_data_filtered = pd.read_csv(label_file_path, header=None, skiprows=range(0, 2826), usecols=[0, 4], sep="\t")
label_data_filtered.columns = ["Sample_ID", "CMS_Label"]
label_data_filtered = label_data_filtered[label_data_filtered['Sample_ID'].str.startswith('TCGA-')]
label_data_filtered['Sample_ID'] = label_data_filtered['Sample_ID'].apply(lambda x: '-'.join(x.split('-')[0:3]))

# Preparing the proteomic data for merging
proteomic_data_for_merge = proteomic_data.transpose()
proteomic_data_for_merge['Sample_ID'] = proteomic_data_for_merge.index
proteomic_data_for_merge['Sample_ID'] = proteomic_data_for_merge['Sample_ID'].apply(lambda x: '-'.join(x.split('-')[0:3]))

# Merging the label data with the proteomic data
merged_data = pd.merge(proteomic_data_for_merge, label_data_filtered, on="Sample_ID", how="left")

# Splitting the data based on CMS labels
cms1_data = merged_data[merged_data['CMS_Label'] == 'CMS1'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()
cms2_data = merged_data[merged_data['CMS_Label'] == 'CMS2'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()
cms3_data = merged_data[merged_data['CMS_Label'] == 'CMS3'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()
cms4_data = merged_data[merged_data['CMS_Label'] == 'CMS4'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()

# Place column 'CMS_label as first column in merged data
merged_data = merged_data[['CMS_Label'] + [col for col in merged_data.columns if col != 'CMS_Label']]
# Combined labelled file
merged_data.set_index('Sample_ID').transpose().to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_ALL_labelled.csv")

# Saving the new dataframes as separate .csv files
cms1_data.to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS1.csv")
cms2_data.to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS2.csv")
cms3_data.to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS3.csv")
cms4_data.to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS4.csv")

total = cms1_data.shape[0] + cms2_data.shape[0] + cms3_data.shape[0] + cms4_data.shape[0]


# # print respective shapes
# print(cms1_data.shape)
# print(cms2_data.shape)
# print(cms3_data.shape)
# print(cms4_data.shape)

# print(proteomic_data.transpose().shape)

# %%
# Drop columns where CMS labels are NaN
filtered_data = merged_data.transpose().dropna(subset=['CMS_Label'], axis=1)

filtered_data = filtered_data.loc[:, ~filtered_data.loc['CMS_Label'].isin(['NOLBL'])]


# Separate the CMS labels and the protein abundance data
cms_labels_filtered = filtered_data.loc['CMS_Label']
protein_data_filtered = filtered_data.drop('CMS_Label').apply(pd.to_numeric, errors='coerce')

# Check the shape to ensure they are compatible
shape_filtered = (cms_labels_filtered.shape, protein_data_filtered.shape)
print(shape_filtered)

# write both to files
cms_labels_filtered.to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/labels_protes.csv")
protein_data_filtered.to_csv("../data/Synapse/TCGA/Proteomics_CMS_groups/protes_only.csv")


# %%
