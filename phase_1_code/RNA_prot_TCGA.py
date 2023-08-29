# %%
import pandas as pd
import numpy as np


proteomic_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/TGCA_Guinney_data/TCGACRC_proteomics.csv"
rna_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/TGCA_Guinney_data/TCGACRC_expression-hi.tsv"

# Reading the proteomic data (CSV format) and setting "GeneSymbol" as the index
proteomic_data = pd.read_csv(proteomic_file_path)
proteomic_data.set_index("GeneSymbol", inplace=True)

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
rna_data

# %%
proteomic_data

# %%
