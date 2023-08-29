### The entries in the Linkedomics1 folder correspond to the CPTAC-2 study

# %%
import pandas as pd
import numpy as np

# Define file paths

proteomic_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/Linkedomics1/proteome_tumornormal.tsv"
rna_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/Linkedomics1/Human__CPTAC_COAD__UNC__RNAseq__HiSeq_RNA__03_01_2017__BCM__Gene__BCM_RSEM_UpperQuartile_log2.cct"

# Read proteomic data (TSV format)
proteomic_data = pd.read_csv(proteomic_file_path, sep="\t")
proteomic_data.set_index("attrib_name", inplace=True)  # Setting protein names as index

# Read RNA data (CCT format)
rna_data = pd.read_csv(rna_file_path, sep="\t")
rna_data.set_index("attrib_name", inplace=True)  # Setting gene names as index

## Print first few rows of both dataframes
# print(proteomic_data.head())
# print(rna_data.head())


# find matching columns with the same name in both dataframes
common_columns = proteomic_data.columns.intersection(rna_data.columns)

# filter the dataframes to only include the matching columns
proteomic_data_match = proteomic_data[common_columns].copy()
rna_data_match = rna_data[common_columns].copy()
order_is_same = all(
    common_columns == proteomic_data.columns.intersection(rna_data.columns))
print(order_is_same)

# Replace all zeros with NaN
proteomic_data_match.replace(0, np.nan, inplace=True)
rna_data_match.replace(0, np.nan, inplace=True)
# Drop all rows containing NaN
proteomic_data_match.dropna(how='any', inplace=True)
rna_data_match.dropna(how='any', inplace=True)



# Concatenate dataframes, aligned by column names
multi_omic_df = pd.concat([proteomic_data, rna_data], axis=0, join="inner")


# %%
# Extract Gene and protein identifiers
rna_data.index
print(f"no of proteins: {len(proteomic_data.index)}")

# Write indeces to 2 separate .txt files
with open("../data/KnowledgeGraphDBs/rna_indeces.tsv", "w") as f:
    for item in rna_data.index:
        f.write("%s\n" % item)

with open("../data/KnowledgeGraphDBs/protein_indeces.tsv", "w") as f2:
    for item in proteomic_data.index:
        f2.write("%s\n" % item)


# Find matching indeces
matching_indeces = list(set(proteomic_data.index).intersection(rna_data.index))
print(len(matching_indeces))
# write matching indeces to a .txt file
with open("../data/KnowledgeGraphDBs/matching_indeces.txt", "w") as f3:
    for item in matching_indeces:
        f3.write("%s\n" % item)


# %%
# Find protein indeces that are not in the RNA data
prot_uniq_indeces= list(set(proteomic_data.index).difference(rna_data.index))
print(len(prot_uniq_indeces))
#write to .txt file
with open("../data/KnowledgeGraphDBs/prot_uniq_indeces.txt", "w") as f4:
    for item in prot_uniq_indeces:
        f4.write("%s\n" % item)

# Find RNA indeces that are not in the protein data
rna_uniq_indeces= list(set(rna_data.index).difference(proteomic_data.index))
print(rna_uniq_indeces)
#write to .txt file
with open("../data/KnowledgeGraphDBs/rna_uniq_indeces.txt", "w") as f5:
    for item in rna_uniq_indeces:
        f5.write("%s\n" % item)

# %%
