# %% 
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import glob

import pandas as pd
import zipfile
from tqdm import tqdm
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats




# %% DATA PIPELINE FOR MAPPING THE ANTIBODY NAMES TO GENE NAMES FROM L4 DATA
# Re-importing pandas as the code execution state was reset
import pandas as pd

# Load the provided files
l4_protein_names_path = '../data/TCGA-COAD-L4/tmp/metadata/L4_protein_names.csv'
xlsx_protein_names_path = '../data/TCGA-COAD-L4/tmp/metadata/xlsx_protein_names.csv'

# Load the provided files
l4_protein_names = pd.read_csv(l4_protein_names_path)
xlsx_protein_names = pd.read_csv(xlsx_protein_names_path)

def find_partial_matches(shorter_names_set, longer_names_set):
    partial_matches = []
    for short_name in shorter_names_set:
        for long_name in longer_names_set:
            if long_name.startswith(short_name):
                partial_matches.append((short_name, long_name))
                break  # Assuming you want only the first match for each short name
    return partial_matches

# # Find direct matches first (without cleaning)
direct_matches = set(l4_protein_names.iloc[:, 0]).intersection(set(xlsx_protein_names.iloc[:, 0]))

# Cleaning names for the next step of matching
l4_protein_names_cleaned = l4_protein_names.iloc[:, 0].str.upper().replace(r'[-_ ()]', '', regex=True)
xlsx_protein_names_cleaned = xlsx_protein_names.iloc[:, 0].str.upper().replace(r'[-_ ()]', '', regex=True)


# Find matches after cleaning
cleaned_matches = set(l4_protein_names_cleaned).intersection(set(xlsx_protein_names_cleaned))

# Combine both sets of matches
total_matches = direct_matches.union(cleaned_matches)

# Separating non-matched names for further analysis
l4_non_matched = set(l4_protein_names_cleaned) - total_matches
xlsx_non_matched = set(xlsx_protein_names_cleaned) - total_matches

# Finding partial matches for the remaining non-matched names
partial_matches_from_l4 = find_partial_matches(l4_non_matched, xlsx_non_matched)
partial_matches_from_xlsx = find_partial_matches(xlsx_non_matched, l4_non_matched)

# Add partial matches to the total matches
total_matches_with_partials = total_matches.union(set([pair[0] for pair in partial_matches_from_l4])).union(set([pair[1] for pair in partial_matches_from_xlsx]))

# Modifying the mapping function to handle cases where a direct match might not be found
def map_cleaned_to_original_with_fallback(cleaned_names, original_names):
    mapping = {}
    original_names_upper = original_names.str.upper().replace(r'[-_ ()]', '', regex=True).tolist()
    for cleaned_name in cleaned_names:
        if cleaned_name in original_names_upper:
            # Find the index of the first occurrence of the cleaned name in the original_names_upper list
            index = original_names_upper.index(cleaned_name)
            mapping[cleaned_name] = original_names.iloc[index]
        else:
            # If no direct match, use the cleaned name as a fallback
            mapping[cleaned_name] = cleaned_name
    return mapping

# Remapping with the modified function
l4_mapping_total = map_cleaned_to_original_with_fallback(total_matches_with_partials, l4_protein_names.iloc[:, 0])
xlsx_mapping_total = map_cleaned_to_original_with_fallback(total_matches_with_partials, xlsx_protein_names.iloc[:, 0])

# Create a DataFrame to display all the matches with the modified mapping function
all_matched_pairs = pd.DataFrame({
    'L4 Original Name': [l4_mapping_total[name] for name in total_matches_with_partials],
    'XLSX Original Name': [xlsx_mapping_total[name] for name in total_matches_with_partials]
})

# # print the names that were not matched
# print('L4 non-matched names:')
# print(l4_non_matched)
# print('XLSX non-matched names:')
# print(xlsx_non_matched)

# # write to csv
all_matched_pairs.to_csv('../data/TCGA-COAD-L4/tmp/metadata/TRUE_L4_MAPPING.csv', index=False)

# Display the first few rows of the all matched pairs DataFrame
all_matched_pairs.shape

# %%
# Load RPPA DATA
RPPA_filename = '../data/TCGA-COAD-L4/tmp/TCGA-COAD-L4-transposed.csv'
RPPA_data = pd.read_csv(RPPA_filename, index_col=0)

# File paths
file_1 = '../data/TCGA-COAD-L4/tmp/metadata/41592_2013_BFnmeth2650_MOESM330_ESM.csv'
file_2 = '../data/TCGA-COAD-L4/tmp/metadata/TRUE_L4_MAPPING.csv'

# Function to add 'Gene Name' column to TRUE_L4_MAPPING.csv using the correspondence in the first file
def add_gene_name_column(file_1_path, file_2_path):
    # Read the CSV files
    df1 = pd.read_csv(file_1_path)
    df2 = pd.read_csv(file_2_path)

    # Mapping 'Protein Name' from df1 to 'Gene Name'
    protein_to_gene_map = dict(zip(df1['Protein Name'], df1['Gene Name']))

    # Creating a new column 'Gene Name' in df2 based on the mapping
    df2['Gene Name'] = df2['XLSX Original Name'].map(protein_to_gene_map)

    return df2

# Adding 'Gene Name' column
updated_df2 = add_gene_name_column(file_1, file_2)

# !!!THIS REMOVES AKT2 AND AKT3 split entries of 'Gene Name' column at spaces and take the first word
updated_df2['Gene Name'] = updated_df2['Gene Name'].str.split().str[0]

# # check overlap between updated_df2['L4 Original Name'] and RPPA_data['Sample_ID'] column
overlap = set(updated_df2['L4 Original Name']).intersection(set(RPPA_data.index))
print(len(overlap))

# Creating a mapping from 'L4 Original Name' to 'Gene Name' in updated_df2
l4_to_gene_map = dict(zip(updated_df2['L4 Original Name'], updated_df2['Gene Name']))

# Adding a 'Gene Name' column to RPPA_data using the mapping, aligning with the corresponding indexes
RPPA_data['Gene Name'] = RPPA_data.index.map(l4_to_gene_map)

# Displaying the updated RPPA_data head
RPPA_data['Gene Name'].head(30)
# Make the ['AB Name'] column from the index
RPPA_data['AB Name'] = RPPA_data.index
# Set 'Gene Name' as the index
RPPA_data.set_index('Gene Name', inplace=True)
# convert to numeric
RPPA_data = RPPA_data.apply(pd.to_numeric, errors='coerce')
# # average values for duplicate indices
RPPA_data = RPPA_data.groupby(RPPA_data.index).mean()

# write to csv

# Split columns on '-', keep first 4 items and rejoin with '-'
RPPA_data.columns = RPPA_data.columns.str.split('-').str[:4].str.join('-')

RPPA_data.to_csv('../data/TCGA-COAD-L4/tmp/RPPA_data_with_gene_name.csv')

RPPA_data.columns








# %% DATA PIPELINE FOR SELECTING THE RNA FILES FROM FOLDERS
# Setup logging
logging.basicConfig(filename='../data/data_processing.log', level=logging.DEBUG)

# Define file paths
sample_sheet_path = '../data/gdc_sample_sheet_850_samples.tsv'
zipped_folder_path = '../data/850_P_R_samples.zip'

# Load the sample sheet
sample_sheet = pd.read_csv(sample_sheet_path, sep='\t')

# Filter the sample sheet to include only 'Gene Expression Quantification' entries
gene_expression_sample_sheet = sample_sheet[sample_sheet['Data Type'] == 'Gene Expression Quantification']

# Extract file IDs from the filtered sample sheet
file_ids = set(gene_expression_sample_sheet['File ID'])

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
file_id_to_sample_id = gene_expression_sample_sheet.set_index('File ID')['Sample ID'].to_dict()

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

# remove rows with all 0s
aggregated_data = aggregated_data.loc[(aggregated_data!=0).any(axis=1)]

# write to file
aggregated_data.to_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA.csv')


# %% 
# Load aggregated data
aggregated_data = pd.read_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA.csv', index_col=0)

# print value of aggregated data at row index 'ACACA' and sample index 'TCGA-A6-6780-01A'
print(aggregated_data.loc['ACACA', 'TCGA-A6-6780-01A'])

# center and scale, row-wise
aggregated_data = aggregated_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

print(aggregated_data.loc['ACACA', 'TCGA-A6-6780-01A'])

# %%
# remove first 4 rows
aggregated_data = aggregated_data.iloc[4:, :]

# remove duplicates by averaging
aggregated_data = aggregated_data.groupby(aggregated_data.index).mean()

# remove rows with all 0s or all NaNs
aggregated_data = aggregated_data.loc[(aggregated_data!=0).any(axis=1)]
aggregated_data = aggregated_data.dropna(axis=0, how='all')

#print value of row with index 'ACACA'
print(aggregated_data.loc['ACACA'])

# write to file
aggregated_data.to_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA_scaled.tsv', sep='\t')



# %% TRANSPOSING THE LABELS FILE
# THIS HAS ALREADY BEEN DONE. IF WE TRANSPOSE AGAIN, IT WILL FLIP BACK. CAUTION
# file_classifier_modified = '../data/TCGA-COAD-L4/tmp/TCGACRC_CMS_CLASSIFIER_LABELS.tsv'


# # THIS IS ONLY NECESSARY IF WE PIPE OUTPUT FROM CMSClassifier.py TO A FILE
# # # transpose file_classifier_modified
# # classifier_data_modified = pd.read_csv(file_classifier_modified, sep='\t', index_col=0).T
# # #set index to col 0
# # classifier_data_modified.index.name = 'Label'

# # # remove rows with 'SSP.median', SSP.min', 'SSP.max' in the index
# # classifier_data_modified = classifier_data_modified.loc[~classifier_data_modified.index.str.contains('SSP.median')]
# # classifier_data_modified = classifier_data_modified.loc[~classifier_data_modified.index.str.contains('SSP.min')]
# # classifier_data_modified = classifier_data_modified.loc[~classifier_data_modified.index.str.contains('SSP.max')]
# # classifier_data_modified = classifier_data_modified.loc[~classifier_data_modified.index.str.contains('SSP.nearestCMS')]

# # SOME COLUMN NAMES FROM THE CLASIFIER HAD AN EXTRA '-1' AT THE END
# # split columns on '-' and keep first 4 items and rejoin with '-'
# classifier_data_modified.columns = classifier_data_modified.columns.str.split('-').str[:4].str.join('-')


# classifier_data_modified.to_csv('../data/TCGA-COAD-L4/tmp/TCGACRC_CMS_CLASSIFIER_LABELS_SLIM.tsv', sep='\t')



# %% APPLYING LABELS TO RPPA DATA
labels_df = pd.read_csv('../data/TCGA-COAD-L4/tmp/TCGACRC_CMS_CLASSIFIER_LABELS_SLIM.tsv', sep='\t')
rppa_df = pd.read_csv('../data/TCGA-COAD-L4/tmp/RPPA_data_with_gene_name.csv')
rna_df = pd.read_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA_scaled.tsv', sep='\t')


def add_labels_to_expression(expression_df, labels_df):
    """
    Correctly add labels as the first row of the RPPA data.

    :param rppa_df: DataFrame with genes in rows and sample IDs in columns (RPPA data).
    :param labels_df: DataFrame with one row containing labels for each sample ID.
    :return: DataFrame with the same format as rppa_df but with an additional first row for labels.
    """
    # Transpose the labels dataframe to match the format of the RPPA data
    labels_transposed = labels_df.transpose()

    # Create a new DataFrame for labels with the same columns as the RPPA data
    labels_row = pd.DataFrame(columns=expression_df.columns)
    
    # Fill the new DataFrame with labels, aligning the columns
    for col in labels_row.columns:
        if col in labels_transposed.index:
            labels_row.at[0, col] = labels_transposed.loc[col].values[0]
        else:
            labels_row.at[0, col] = None

    # Concatenate the labels row with the RPPA data
    rppa_with_labels = pd.concat([labels_row, expression_df], ignore_index=True)

    # set fist column as index
    rppa_with_labels.set_index(rppa_with_labels.columns[0], inplace=True)

    return rppa_with_labels

# Apply the function to the loaded data
rppa_with_labels = add_labels_to_expression(rppa_df, labels_df)

rna_with_labels = add_labels_to_expression(rna_df, labels_df)

# %%

# MATCHING VARIABLES BETWEEN RNA AND PROT
# print value of the 'TCGA-A6-6780-01A' column, at row index 'ACACA'

# THIS MATCHES THE RPPA VARIABLES TO THE RNA AND VICE VERSA
# # select rows in aggregated_data that are also in RPPA_data
# rna_with_labels = rna_with_labels.loc[rna_with_labels.index.isin(rppa_with_labels.index)]

# # select rows in RPPA_data that are also in aggregated_data
# rppa_with_labels = rppa_with_labels.loc[rppa_with_labels.index.isin(rna_with_labels.index)]

# keep only columns with 'CMS1', 'CMS2', 'CMS3' in the first row
rna_with_labels_123 = rna_with_labels.loc[:, rna_with_labels.iloc[0].isin(['CMS1', 'CMS2', 'CMS3'])]
rppa_with_labels_123 = rppa_with_labels.loc[:, rppa_with_labels.iloc[0].isin(['CMS1', 'CMS2', 'CMS3'])]

# transpose all dataframes
rna_with_labels_123T = rna_with_labels_123.transpose()
rppa_with_labels_123T = rppa_with_labels_123.transpose()

rna_with_labelsT = rna_with_labels.transpose()
rppa_with_labelsT = rppa_with_labels.transpose()

# # THIS COLUMN WAS DUPLICATE
# # print values of row with index 'TCGA-A6-6780-01A'
# print(rna_with_labelsT.loc['TCGA-A6-6780-01A'])



# write to file
rna_with_labels_123T.to_csv('../data/TCGA-COAD-L4/tmp/RNA_for_RPPA_scaled_labels_123T.csv')
rppa_with_labels_123T.to_csv('../data/TCGA-COAD-L4/tmp/RPPA_gene_name_labels_123T.csv')

rna_with_labelsT.to_csv('../data/TCGA-COAD-L4/tmp/RNA_for_RPPA_scaled_labels_ALLT.csv')
rppa_with_labelsT.to_csv('../data/TCGA-COAD-L4/tmp/RPPA_gene_name_labels_ALLT.csv')



# %% CHECK TO SEE IF EXPRESSION FROM GUINNEY DATA MATCHES THE RPPA DATA

expression_file = '../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_ALL_labelled.csv'
rppa_file = '../data/TCGA-COAD-L4/tmp/RPPA_gene_name_labels_ALLT.csv'


expression_df = pd.read_csv(expression_file, index_col=0)
rppa_df = pd.read_csv(rppa_file, index_col=0)

expression_df_123 = expression_df.loc[expression_df.iloc[:, 0].isin(['CMS1', 'CMS2', 'CMS3'])]
rppa_df_123 = rppa_df.loc[rppa_df.iloc[:, 0].isin(['CMS1', 'CMS2', 'CMS3'])]

# check overlap between expression_df.columns and rppa_df.columns
overlap = set(expression_df.columns).intersection(set(rppa_df.columns))
print('total number of variables in rppa: {}'.format(len(rppa_df.columns)))
print(f'variables both in guinney expression and rppa: {len(overlap)}')

# split index of rppa_df on '-' and keep first 3 items and rejoin with '-'
rppa_df.index = rppa_df.index.str.split('-').str[:3].str.join('-')

# keep only columns that are in ovlerlap
rppa_df = rppa_df.loc[:, rppa_df.columns.isin(overlap)]
expression_df = expression_df.loc[:, expression_df.columns.isin(overlap)]

rppa_df_123 = rppa_df_123.loc[:, rppa_df_123.columns.isin(overlap)]
expression_df_123 = expression_df_123.loc[:, expression_df_123.columns.isin(overlap)]


# center and scale across columns
rppa_df = rppa_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
expression_df = expression_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

rppa_df_123 = rppa_df_123.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
expression_df_123 = expression_df_123.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# set random seed
# select random column from rppa_df as data
data = rppa_df.iloc[:, np.random.randint(0, rppa_df.shape[1])]

# Generate QQ-plot
(fig, ax) = plt.subplots()
stats.probplot(data, dist="norm", plot=ax)
ax.get_lines()[1].set_color('r')  # Optional: change the color of the QQ-plot line

# # Add 45-degree line for reference
# ax.plot([np.min(data), np.max(data)], [np.min(data), np.max(data)], 'k--')

plt.title('QQ-plot with Reference Line')
plt.show()

# check shape of all dataframes
print(f'expression_df shape: {expression_df.shape}')
print(f'rppa_df shape: {rppa_df.shape}')
print(f'expression_df_123 shape: {expression_df_123.shape}')
print(f'rppa_df_123 shape: {rppa_df_123.shape}')


# write to csv
rppa_df.to_csv('../Diffusion/data/RPPA_for_pig_ALL.csv')
expression_df.to_csv('../Diffusion/data/Expression_for_pig_ALL.csv')

rppa_df_123.to_csv('../Diffusion/data/RPPA_for_pig_123.csv')
expression_df_123.to_csv('../Diffusion/data/Expression_for_pig_123.csv')


















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




