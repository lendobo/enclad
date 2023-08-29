# %%

import pandas as pd

# Reading in the microarray file (assuming tab-separated values)
microarray_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/TGCA_CMS_data/TCGACRC_microarray.txt"
microarray_data = pd.read_csv(microarray_file_path, sep="\t")

# Reading in the proteomics file (assuming comma-separated values)
proteomics_file_path = "/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/Codebase2/data/TGCA_CMS_data/TCGACRC_proteomics.csv"
proteomics_data = pd.read_csv(proteomics_file_path)

# Checking the first few rows of each dataframe to understand the structure
microarray_data.head()

# proteomics_data.head()

# %%
# find matching columns
matching_columns = list(
    set(microarray_data.columns).intersection(proteomics_data.columns)
)

# filter the dataframes to only include the matching columns
microarray_data_match = microarray_data[matching_columns]
proteomics_data_match = proteomics_data[matching_columns]

# %%
microarray_data_match.head()
