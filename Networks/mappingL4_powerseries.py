# %% 
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import glob




# %%
# Load the provided files
L4_rppa_file_path = '../data/TCGA-COAD-L4/tmp/L4_genenames.tsv'
L4_mapping = '../data/TCGA-COAD-L4/tmp/mapped_antibodiy_togene.csv'
bigfolder_rppa_file_path = '../data/350_PROT_samples/matched_to_rna/prot_rppa_names.csv'
rna_file_path = '../data/Synapse/TCGA/RNA_CMS_groups/RNA_names.csv'
mass_spec = '../data/Synapse/TCGA/Proteomics_CMS_groups/Prot_Names.csv'

# Reading the data from the files
L4_rppa_data = pd.read_csv(L4_rppa_file_path, sep='\t')
L4_rppa_data['L4_Gene_Symbol_RPPA'] = L4_rppa_data.columns
L4_rppa_data = L4_rppa_data[['L4_Gene_Symbol_RPPA']]
L4_mapping = pd.read_csv(L4_mapping)
rppa_data = pd.read_csv(bigfolder_rppa_file_path)
rna_data = pd.read_csv(rna_file_path)
mass_spec_data = pd.read_csv(mass_spec)

print(L4_rppa_data.shape, rppa_data.shape, rna_data.shape)
# Display the first few rows of each dataset to understand their structure
L4_rppa_data.head(), L4_mapping.head(), rppa_data.head(), rna_data.head()


# %%
# Renaming the columns for easier understanding
rppa_data.columns = ['Gene_Symbol_RPPA']
rna_data.columns = ['Gene_Symbol_RNA']
L4_mapping.columns = ['Gene_Symbol_RPPA']
mass_spec_data.columns = ['Gene_Symbol_RPPA']

# Convert all to UPPERCASE for consistency
L4_rppa_data['L4_Gene_Symbol_RPPA'] = L4_rppa_data['L4_Gene_Symbol_RPPA'].str.upper()
rppa_data['Gene_Symbol_RPPA'] = rppa_data['Gene_Symbol_RPPA'].str.upper()
rna_data['Gene_Symbol_RNA'] = rna_data['Gene_Symbol_RNA'].str.upper()
L4_mapping['Gene_Symbol_RPPA'] = L4_mapping['Gene_Symbol_RPPA'].str.upper()
mass_spec_data['Gene_Symbol_RPPA'] = mass_spec_data['Gene_Symbol_RPPA'].str.upper()


# Finding the common elements
common_genes_rppa = pd.merge(rppa_data, rna_data, left_on='Gene_Symbol_RPPA', right_on='Gene_Symbol_RNA', how='inner')
common_genes_L4 = pd.merge(L4_rppa_data, rna_data, left_on='L4_Gene_Symbol_RPPA', right_on='Gene_Symbol_RNA', how='inner')
common_genes_L4_rppa = pd.merge(L4_rppa_data, rppa_data, left_on='L4_Gene_Symbol_RPPA', right_on='Gene_Symbol_RPPA', how='inner')
common_genes_mapping_rna = pd.merge(L4_mapping, rna_data, left_on='Gene_Symbol_RPPA', right_on='Gene_Symbol_RNA', how='inner')

# check overlap between mass_spec and common_genes_mapping_rna
common_genes_mass_spec_L4_rna = pd.merge(mass_spec_data, common_genes_mapping_rna, left_on='Gene_Symbol_RPPA', right_on='Gene_Symbol_RPPA', how='inner')
common_genes_mass_spec_bigfolder_rna = pd.merge(mass_spec_data, common_genes_rppa, left_on='Gene_Symbol_RPPA', right_on='Gene_Symbol_RPPA', how='inner')

# # Displaying the common genes
# common_genes_rppa.head(), len(common_genes_rppa)
# common_genes_L4.head(), len(common_genes_L4)
# common_genes_L4_rppa.head(), len(common_genes_L4_rppa)

print(f'Common genes between mass spec, mapped L4 and rna: {len(common_genes_mass_spec_L4_rna)}')
print(f'Common genes between mass spec, mapped big folder and rna: {len(common_genes_mass_spec_bigfolder_rna)}\n')

print('Number of genes in the big folder: ', len(rppa_data))
print('Number of common genes between big folder and RNA: ', len(common_genes_rppa), '\n')

print('L4 data, unmapped: ', len(L4_rppa_data))
print('Mapped antibodies to genes: ', len(L4_mapping))
print('Common genes between L4 mapping and RNA: ', len(common_genes_mapping_rna))
common_genes_mapping_rna.head(), len(common_genes_mapping_rna)

# write the common genes to a csv file
common_genes_rppa['Gene_Symbol_RPPA'].to_csv('../data/350_PROT_samples/matched_to_rna/common_genes.csv', index=False)
common_genes_mapping_rna['Gene_Symbol_RPPA'].to_csv('../data/TCGA-COAD-L4/tmp/common_genes_L4_RNA.csv', index=False)




# %% PARSING THE BIGFOLDER
import os
import re
import pandas as pd

# Path to the directory containing all the folders
directory_path = '../data/350_PROT_samples'

# Regular expression to match the 'TCGA-XX-XXXX' pattern
pattern = r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}'

# List to store the extracted patterns
tcga_patterns = []

# Loop through each subfolder and file in the directory
for folder_name in os.listdir(directory_path):
    subfolder_path = os.path.join(directory_path, folder_name)
    if os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.startswith('TCGA'):
                match = re.search(pattern, file_name)
                if match:
                    tcga_patterns.append(match.group())

# Creating a pandas DataFrame
bigfolder_df = pd.DataFrame(tcga_patterns, columns=['Sample_ID'])

# Display the DataFrame (optional)
print(bigfolder_df)



# %% FOLDER MATCHING
sample_sheet = '../data/500_RNA_samples/gdc_sample_sheet.2023-11-29.tsv'
sample_sheet_df = pd.read_csv(sample_sheet, sep='\t')
sample_sheet_df.head()


# %%
samples_L4 = '../data/TCGA-COAD-L4/tmp/metadata/sampleIDs_L4.csv'
samples_RNA = '../data/Synapse/TCGA/RNA_CMS_groups/sampleIDs_RNA.tsv'

samples_L4_df = pd.read_csv(samples_L4)
samples_RNA_df = pd.read_csv(samples_RNA, sep='\t')

samples_RNA_df['Sample_ID'] = samples_RNA_df.columns
samples_RNA_df = samples_RNA_df[['Sample_ID']]



# for each value in the 'Sample_ID' column, split at the 3rd hyphen and take the first part
samples_L4_df['Sample_ID'] = samples_L4_df['Sample_ID'].apply(lambda x: x.split('-')[:4])
# concatenate the split values into a single string
samples_L4_df['Sample_ID'] = samples_L4_df['Sample_ID'].apply(lambda x: '-'.join(x))

samples_L4_df.head()

# check overlap between L4 and sample_sheet
common_samples_L4_rna = pd.merge(samples_L4_df, sample_sheet_df, left_on='Sample_ID', right_on='Sample ID', how='inner')
common_samples_L4_rna.head()

# len(common_samples_L4_rna)

# common_samples_L4.to_csv('../data/TCGA-COAD-L4/tmp/common_samples_L4_RNA.csv', index=False)







# %% DATA PIPELINE
import pandas as pd
import zipfile
from tqdm import tqdm

# Modified code to include batching and error logging for processing large number of files

import pandas as pd
import zipfile
import logging

# Setup logging
logging.basicConfig(filename='../data/data_processing.log', level=logging.DEBUG)

# Define file paths
sample_sheet_path = '../data/gdc_sample_sheet_850_samples.tsv'
common_samples_path = '../data/TCGA-COAD-L4/tmp/metadata/common_samples_L4_RNA.csv'
zipped_folder_path = '../data/850_P_R_samples.zip'

# Load the sample sheet and the common samples file
sample_sheet = pd.read_csv(sample_sheet_path, sep='\t')
common_samples = pd.read_csv(common_samples_path)

# Filter the sample sheet to include only 'Gene Expression Quantification' entries
gene_expression_sample_sheet = sample_sheet[sample_sheet['Data Type'] == 'Gene Expression Quantification']
filtered_sample_sheet = gene_expression_sample_sheet[gene_expression_sample_sheet['Sample ID'].isin(common_samples['Sample_ID'])]

# Extract file IDs from the filtered sample sheet
file_ids = set(filtered_sample_sheet['File ID'])

# Initialize a dictionary to store the gene data
gene_data = {}

# Process each file in the zip archive with batching
batch_size = 50
current_batch = 0

try:
    with zipfile.ZipFile(zipped_folder_path, 'r') as zipped_folder:
        all_files = [file for file in zipped_folder.namelist() if file.endswith('.tsv')]
        
        # Process files in batches
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]
            current_batch += 1

            for file in batch_files:
                folder_name = file.split('/')[1]  # Adjusted to match the observed folder structure

                if folder_name in file_ids:
                    # Extract and read the .tsv file
                    with zipped_folder.open(file) as tsv_file:
                        # Skip the first line (comment) and read the second line for headers
                        next(tsv_file)
                        tsv_df = pd.read_csv(tsv_file, sep='\t', comment='#')

                        # Select only 'gene_name' and 'unstranded' columns
                        if 'gene_name' in tsv_df.columns and 'unstranded' in tsv_df.columns:
                            gene_data[folder_name] = tsv_df[['gene_name', 'unstranded']].set_index('gene_name')

            logging.info(f"Processed batch {current_batch} containing {len(batch_files)} files.")

except Exception as e:
    logging.error(f"Error occurred during batch {current_batch}: {e}")

# Create an empty DataFrame for aggregating the data
aggregated_data = pd.DataFrame()

# Map File IDs to Sample IDs from the sample sheet
file_id_to_sample_id = filtered_sample_sheet.set_index('File ID')['Sample ID'].to_dict()

# Aggregate the data
for file_id, data in gene_data.items():
    data_clean = data.dropna().copy()
    sample_id = file_id_to_sample_id.get(file_id, None)

    if sample_id:
        data_clean.rename(columns={'unstranded': sample_id}, inplace=True)
        aggregated_data = pd.concat([aggregated_data, data_clean], axis=1)

# Path for logging file
log_file_path = '../data/data_processing.log'

# %%
aggregated_data.shape






















# %%

t = np.arange(0, 2, 0.1)
k = np.arange(0, 10, 1)

def laplacian_exponential(t, k):
    return np.sum([t**ki / np.math.factorial(ki) for ki in range(k+1)])

# create a multiplot of size (k/2) * 5
fig, axs = plt.subplots(int(len(k)/2), 2, figsize=(10, 20))
for i in range(len(k)):
    axs[int(i/2), i%2].plot(t, [laplacian_exponential(ti, k[i]) for ti in t])
    axs[int(i/2), i%2].set_title('k = {}'.format(k[i]))
    axs[int(i/2), i%2].set(xlabel='t', ylabel='Laplacian exponential')
    axs[int(i/2), i%2].grid()


plt.tight_layout()
plt.show()
# %%

k = [2,5]

short_distance = []
long_distance = []

t = np.linspace(0.1, 6, 10)

ki = 2
for ti in t:
    short_distance.append(ti**ki / np.math.factorial(ki))


ki = 10
for ti in t:
    long_distance.append(ti**ki / np.math.factorial(ki))


plt.plot(t, short_distance, label='k = 2')
plt.plot(t, long_distance, label='k = 5')
plt.legend()
plt.show()


# %%
# create string graph
import networkx as nx

G = nx.Graph()
G.add_nodes_from([1,2,3,4,5])
G.add_edges_from([(1,2), (2,3), (3,4), (4,5)])

nx.draw(G, with_labels=True)    
plt.show()

# Get laplacian matrix
L = -(nx.laplacian_matrix(G).todense())
print(L)
print(L@L)
print(L@L@L)
print(L@L@L@L)




