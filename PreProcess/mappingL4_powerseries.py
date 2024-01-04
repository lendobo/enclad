# %% 
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import glob

import zipfile
from tqdm import tqdm
import logging

from scipy import stats
import scipy.stats as stats
from scipy.stats.mstats import winsorize

from sklearn.preprocessing import PowerTransformer

import statsmodels



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
XLSX_mapping_file = '../data/TCGA-COAD-L4/tmp/metadata/41592_2013_BFnmeth2650_MOESM330_ESM.csv'
L4_to_XLSX = '../data/TCGA-COAD-L4/tmp/metadata/TRUE_L4_MAPPING.csv'

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
L4_to_XLSX_PROT_to_GENE = add_gene_name_column(XLSX_mapping_file, L4_to_XLSX)

# # print number of unique entries for of the 3 columns separately
# print(len(L4_to_XLSX_PROT_to_GENE['L4 Original Name'].unique()))
# print(len(L4_to_XLSX_PROT_to_GENE['XLSX Original Name'].unique()))
# print(len(L4_to_XLSX_PROT_to_GENE['Gene Name'].unique()))

# !!!THIS REMOVES AKT2 AND AKT3 split entries of 'Gene Name' column at spaces and take the first word
# STRATEGY: If downstream analysis reveals significant genes, check if they share antibodies with other genes (treat collectively)
L4_to_XLSX_PROT_to_GENE['Gene Name'] = L4_to_XLSX_PROT_to_GENE['Gene Name'].str.split().str[0]

# # # check overlap between updated_df2['L4 Original Name'] and RPPA_data['Sample_ID'] column
# overlap = set(L4_to_XLSX_PROT_to_GENE['L4 Original Name']).intersection(set(RPPA_data.index))
# print(len(overlap))

# Creating a mapping from 'L4 Original Name' to 'Gene Name' in updated_df2
l4_to_gene_map = dict(zip(L4_to_XLSX_PROT_to_GENE['L4 Original Name'], L4_to_XLSX_PROT_to_GENE['Gene Name']))

# Adding a 'Gene Name' column to RPPA_data using the mapping, aligning with the corresponding indexes
RPPA_data['Gene Name'] = RPPA_data.index.map(l4_to_gene_map)

# # print length of unique values in 'Gene Name' column
# print(len(RPPA_data['Gene Name'].unique()))


# # Set 'Gene Name' as the index
RPPA_data.set_index('Gene Name', inplace=True)
# convert to numeric
RPPA_data = RPPA_data.apply(pd.to_numeric, errors='coerce')
# # average values for duplicate indices
RPPA_data = RPPA_data.groupby(RPPA_data.index).mean()


# Split columns on '-', keep first 4 items and rejoin with '-'
RPPA_data.columns = RPPA_data.columns.str.split('-').str[:4].str.join('-')

RPPA_data.to_csv('../data/TCGA-COAD-L4/tmp/RPPA_data_with_gene_name.csv')









# # %% DATA PIPELINE FOR SELECTING THE RNA FILES FROM FOLDERS
# # Setup logging
# logging.basicConfig(filename='../data/data_processing.log', level=logging.DEBUG)

# # Define file paths
# sample_sheet_path = '../data/gdc_sample_sheet_850_samples.tsv'
# zipped_folder_path = '../data/850_P_R_samples.zip'

# # Load the sample sheet
# sample_sheet = pd.read_csv(sample_sheet_path, sep='\t')

# # Filter the sample sheet to include only 'Gene Expression Quantification' entries
# gene_expression_sample_sheet = sample_sheet[sample_sheet['Data Type'] == 'Gene Expression Quantification']

# # Extract file IDs from the filtered sample sheet
# file_ids = set(gene_expression_sample_sheet['File ID'])

# # Initialize a dictionary to store the gene data
# gene_data = {}

# # Process each file in the zip archive with batching
# batch_size = 50
# current_batch = 0

# try:
#     with zipfile.ZipFile(zipped_folder_path, 'r') as zipped_folder:
#         all_files = [file for file in zipped_folder.namelist() if file.endswith('.tsv')]
        
#         # Process files in batches
#         for i in range(0, len(all_files), batch_size):
#             batch_files = all_files[i:i + batch_size]
#             current_batch += 1

#             for file in batch_files:
#                 folder_name = file.split('/')[1]  # Adjusted to match the observed folder structure

#                 if folder_name in file_ids:
#                     # Extract and read the .tsv file
#                     with zipped_folder.open(file) as tsv_file:
#                         # Skip the first line (comment) and read the second line for headers
#                         next(tsv_file)
#                         tsv_df = pd.read_csv(tsv_file, sep='\t', comment='#')

#                         # Select only 'gene_name' and 'unstranded' columns
#                         if 'gene_name' in tsv_df.columns and 'unstranded' in tsv_df.columns:
#                             gene_data[folder_name] = tsv_df[['gene_name', 'unstranded']].set_index('gene_name')

#             logging.info(f"Processed batch {current_batch} containing {len(batch_files)} files.")

# except Exception as e:
#     logging.error(f"Error occurred during batch {current_batch}: {e}")

# # Create an empty DataFrame for aggregating the data
# aggregated_data = pd.DataFrame()

# # Map File IDs to Sample IDs from the sample sheet
# file_id_to_sample_id = gene_expression_sample_sheet.set_index('File ID')['Sample ID'].to_dict()

# # Aggregate the data
# for file_id, data in gene_data.items():
#     data_clean = data.dropna().copy()
#     sample_id = file_id_to_sample_id.get(file_id, None)

#     if sample_id:
#         data_clean.rename(columns={'unstranded': sample_id}, inplace=True)
#         aggregated_data = pd.concat([aggregated_data, data_clean], axis=1)

# # Path for logging file
# log_file_path = '../data/data_processing.log'




# # %%
# aggregated_data.shape

# # remove rows with all 0s
# aggregated_data = aggregated_data.loc[(aggregated_data!=0).any(axis=1)]

# # write to file
# aggregated_data.to_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA.csv')


# # %% 
# # Load aggregated data
# aggregated_data = pd.read_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA.csv', index_col=0)

# # print value of aggregated data at row index 'ACACA' and sample index 'TCGA-A6-6780-01A'
# print(aggregated_data.loc['ACACA', 'TCGA-A6-6780-01A'])

# # center and scale, row-wise
# aggregated_data = aggregated_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

# print(aggregated_data.loc['ACACA', 'TCGA-A6-6780-01A'])

# # %%
# # remove first 4 rows
# aggregated_data = aggregated_data.iloc[4:, :]

# # remove duplicates by averaging
# aggregated_data = aggregated_data.groupby(aggregated_data.index).mean()

# # remove rows with all 0s or all NaNs
# aggregated_data = aggregated_data.loc[(aggregated_data!=0).any(axis=1)]
# aggregated_data = aggregated_data.dropna(axis=0, how='all')

# #print value of row with index 'ACACA'
# print(aggregated_data.loc['ACACA'])

# # write to file
# aggregated_data.to_csv('../data/TCGA-COAD-L4/tmp/RNA_samples_for_RPPA_scaled.tsv', sep='\t')



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


def add_labels_to_expression(expr_df, labels_df):
    """
    Correctly add labels as the first row of the RPPA data.

    :param rppa_df: DataFrame with genes in rows and sample IDs in columns (RPPA data).
    :param labels_df: DataFrame with one row containing labels for each sample ID.
    :return: DataFrame with the same format as rppa_df but with an additional first row for labels.
    """
    # Transpose the labels dataframe to match the format of the RPPA data
    labels_transposed = labels_df.transpose()

    # Create a new DataFrame for labels with the same columns as the RPPA data
    labels_row = pd.DataFrame(columns=expr_df.columns)
    
    # Fill the new DataFrame with labels, aligning the columns
    for col in labels_row.columns:
        if col in labels_transposed.index:
            labels_row.at[0, col] = labels_transposed.loc[col].values[0]
        else:
            labels_row.at[0, col] = None

    # Concatenate the labels row with the RPPA data
    rppa_with_labels = pd.concat([labels_row, expr_df], ignore_index=True)

    # set fist column as index
    rppa_with_labels.set_index(rppa_with_labels.columns[0], inplace=True)

    return rppa_with_labels

# Apply the function to the loaded data
rppa_with_labels = add_labels_to_expression(rppa_df, labels_df)

rna_with_labels = add_labels_to_expression(rna_df, labels_df)

rna_with_labels_123 = rna_with_labels.loc[:, rna_with_labels.iloc[0].isin(['CMS1', 'CMS2', 'CMS3'])]
rna_with_labels_4 = rna_with_labels.loc[:, rna_with_labels.iloc[0].isin(['CMS4'])]

rppa_with_labels_123 = rppa_with_labels.loc[:, rppa_with_labels.iloc[0].isin(['CMS1', 'CMS2', 'CMS3'])]
rppa_with_labels_4 = rppa_with_labels.loc[:, rppa_with_labels.iloc[0].isin(['CMS4'])]

# print all shapes
print(f'rna_with_labels_123 shape: {rna_with_labels_123.shape}')
print(f'rna_with_labels_4 shape: {rna_with_labels_4.shape}')
print(f'rppa_with_labels_123 shape: {rppa_with_labels_123.shape}')
print(f'rppa_with_labels_4 shape: {rppa_with_labels_4.shape}')

# %%
# ATTACHING LABELS TO BOTH RPPA NAD RNA DATA

# keep only columns with 'CMS1', 'CMS2', 'CMS3' in the first row
rna_with_labels_123 = rna_with_labels.loc[:, rna_with_labels.iloc[0].isin(['CMS1', 'CMS2', 'CMS3'])]
rppa_with_labels_123 = rppa_with_labels.loc[:, rppa_with_labels.iloc[0].isin(['CMS1', 'CMS2', 'CMS3'])]

# transpose all dataframes
rna_with_labels_123T = rna_with_labels_123.transpose()
rppa_with_labels_123T = rppa_with_labels_123.transpose()

rna_with_labelsT = rna_with_labels.transpose()
rppa_with_labelsT = rppa_with_labels.transpose()


# write to file
rna_with_labels_123T.to_csv('../data/TCGA-COAD-L4/tmp/RNA_for_RPPA_scaled_labels_123T.csv')
rppa_with_labels_123T.to_csv('../data/TCGA-COAD-L4/tmp/RPPA_gene_name_labels_123T.csv')

rna_with_labelsT.to_csv('../data/TCGA-COAD-L4/tmp/RNA_for_RPPA_scaled_labels_ALLT.csv')
rppa_with_labelsT.to_csv('../data/TCGA-COAD-L4/tmp/RPPA_gene_name_labels_ALLT.csv')


















# %% LINKEDOMICS LINKEDOMICS LINKEDOMICS
linked_RNA = pd.read_csv('../data/LinkedOmics/linked_rna.cct', sep='\t', index_col=0)
linked_rppa = pd.read_csv('../data/LinkedOmics/linked_rppa.tsv', sep='\t', index_col=0)

# transpose
linked_RNA_T = linked_RNA.transpose()
linked_rppa_T = linked_rppa.transpose()

classifier_labels = pd.read_csv('../data/LinkedOmics/TCGACRC_CMS_CLASSIFIER_LABELS.tsv', sep='\t')
classifier_labels.rename(columns={classifier_labels.columns[0]: 'sample_ID'}, inplace=True)

# remove all columns except for 'sampleID' and 'SSP.nearestCMS'
classifier_labels = classifier_labels.loc[:, classifier_labels.columns.isin(['sample_ID', 'SSP.nearestCMS'])]

# set index to 'sample_ID'
classifier_labels.set_index('sample_ID', inplace=True)


# Merge the labels with the transposed RNA data
linked_RNA_T_labeled = linked_RNA_T.join(classifier_labels)
# Merge the labels with the transposed RPPA data
linked_rppa_T_labeled = linked_rppa_T.join(classifier_labels)

# only keep columns that are in both dataframes
linked_RNA_T_labeled = linked_RNA_T_labeled.loc[:, linked_RNA_T_labeled.columns.isin(linked_rppa_T_labeled.columns)]
linked_rppa_T_labeled = linked_rppa_T_labeled.loc[:, linked_rppa_T_labeled.columns.isin(linked_RNA_T_labeled.columns)]

# make subset df with only CMS1, CMS2, CMS3 and then drop the last column
linked_RNA_T_123 = linked_RNA_T_labeled.loc[linked_RNA_T_labeled.iloc[:, -1].isin(['CMS1', 'CMS2', 'CMS3'])].drop(columns=['SSP.nearestCMS'])
linked_rppa_T_123 = linked_rppa_T_labeled.loc[linked_rppa_T_labeled.iloc[:, -1].isin(['CMS1', 'CMS2', 'CMS3'])].drop(columns=['SSP.nearestCMS'])
linked_RNA_T_ALL = linked_RNA_T_labeled.drop(columns=['SSP.nearestCMS'])
linked_rppa_T_ALL = linked_rppa_T_labeled.drop(columns=['SSP.nearestCMS'])

# write to csv
linked_RNA_T_ALL.to_csv('../data/LinkedOmics/linked_rna_ALL.csv')
linked_rppa_T_ALL.to_csv('../data/LinkedOmics/linked_rppa_ALL.csv')
linked_RNA_T_123.to_csv('../data/LinkedOmics/linked_rna_123.csv')
linked_rppa_T_123.to_csv('../data/LinkedOmics/linked_rppa_123.csv')

# print shapes
print(f'linked_RNA_T_ALL shape: {linked_RNA_T_ALL.shape}')
print(f'linked_rppa_T_ALL shape: {linked_rppa_T_ALL.shape}')
print(f'linked_RNA_T_123 shape: {linked_RNA_T_123.shape}')
print(f'linked_rppa_T_123 shape: {linked_rppa_T_123.shape}')


# %% CHECK TO SEE IF EXPRESSION FROM GUINNEY DATA MATCHES THE RPPA DATA

def apply_yeo_johnson(column):
    transformer = PowerTransformer(method='yeo-johnson')
    # Reshape data for transformation (needs to be 2D)
    column_reshaped = column.values.reshape(-1, 1)
    transformed_column = transformer.fit_transform(column_reshaped)
    # Flatten the array to 1D
    return transformed_column.flatten()

def filter_dataframes(df1, df2):
    # check overlap between df1.columns and df2.columns
    overlap = set(df1.columns).intersection(set(df2.columns))
    print('total number of variables in df2: {}'.format(len(df2.columns)))
    print(f'variables both in df1 and df2: {len(overlap)}')

    # keep only columns that are in overlap
    df1 = df1.loc[:, df1.columns.isin(overlap)]
    df2 = df2.loc[:, df2.columns.isin(overlap)]

    return df1, df2

def center_and_scale(df, axis=0):
    # center and scale across columns
    df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df


# Load the data
linked_RNA_T_ALL = pd.read_csv('../data/LinkedOmics/linked_rna_ALL.csv', index_col=0)
linked_rppa_T_ALL = pd.read_csv('../data/LinkedOmics/linked_rppa_ALL.csv', index_col=0)
linked_RNA_T_123 = pd.read_csv('../data/LinkedOmics/linked_rna_123.csv', index_col=0)
linked_rppa_T_123 = pd.read_csv('../data/LinkedOmics/linked_rppa_123.csv', index_col=0)


# center and scale across columns
PROT_df_ALL_SC = center_and_scale(linked_rppa_T_ALL)
RNA_df_ALL_SC = center_and_scale(linked_RNA_T_ALL)
PROT_df_123_SC = center_and_scale(linked_rppa_T_123)
RNA_df_123_SC = center_and_scale(linked_RNA_T_123)



def transform_and_test(data_to_trim, dataframe_name):
    print('--------------------------------------------------------------')
    print(f'Initial results for {dataframe_name}')
    
    # Winsorize
    data_to_trim = data_to_trim.apply(lambda x: winsorize(x, limits=[0.01, 0.01]), axis=0)

    alpha = 0.05
    original_ks_results = {}

    # Perform initial K-S test and store results
    for column in data_to_trim.columns:
        data = data_to_trim[column].dropna()
        if data.nunique() > 1 and len(data) > 3:
            stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            original_ks_results[column] = (stat, p)
            if p < alpha:
                # Apply Yeo-Johnson transformation
                data_to_trim[column] = apply_yeo_johnson(data_to_trim[column])

    # Perform K-S test again on transformed columns and compare
    for column, (original_stat, original_p) in original_ks_results.items():
        if original_p < alpha:
            transformed_data = data_to_trim[column].dropna()
            new_stat, new_p = stats.kstest(transformed_data, 'norm', args=(transformed_data.mean(), transformed_data.std()))
            if new_p < alpha:
                print(f'Column: {column}')
                print(f'  Original K-S Statistic: {original_stat}, p-value: {original_p}')
                print(f'  Transformed K-S Statistic: {new_stat}, p-value: {new_p}')
                print('--------------------------------------------------------------\n')

                # make QQ-plot for these columns
                (fig, ax) = plt.subplots()
                stats.probplot(transformed_data, dist="norm", plot=ax)
                ax.set_title(f'QQ-plot for {column} (W: {round(new_stat, 3)}, p: {new_p}')
                plt.show()
    
    return data_to_trim

# Apply transformations and tests to each dataframe
prot_df_all_transformed = transform_and_test(PROT_df_ALL_SC, 'PROT_df_ALL')
rna_df_all_transformed = transform_and_test(RNA_df_ALL_SC, 'RNA_df_ALL')
prot_df_123_transformed = transform_and_test(PROT_df_123_SC, 'PROT_df_123')
rna_df_123_transformed = transform_and_test(RNA_df_123_SC, 'RNA_df_123_SC')

print(prot_df_123_transformed.shape)


whitelist = ['VEGFR2', 'CDH1', 'BRAF', 'BAP1', 'TP53', 'CASP7', 'PRKCD', 'RAB11A', 'YAP1', 'CTNNB1', 'CCNB1', 'CCNE1', 
             'CCNE2', 'HSPA1A', 'ARID1A', 'ASNS', 'CHEK2', 'PCNA', 'ITGA2', 'MAPK1', 'ANXA1', 'CLDN7', 'COL6A1', 'FN1', 
             'MYH11','TP53BP1', 'EIF4EBP1', 'EEF2K', 'EIF4G1', 'FRAP1', 'RICTOR', 'RPS6', 'TSC1', 'RPS6KA1', 'ACACA',
             'AR', 'KIT', 'EGFR', 'FASN', 'ERBB3', 'IGFBP2', 'CDKN1A', 'CDKN1B', 'SQSTM1', 'PEA15', 'RB1', 'ACVRL1'
             'SMAD1', 'FOXM1', 'FOXO3', 'CAV1', 'PARK7', 'SERPINE1', 'RBM15', 'WWTR1', 'TGM2']
blacklist = ['ERBB2', 'NKX2-1', 'RAD50']

# %%

# remove columns in blacklist
prot_df_all_transformed = prot_df_all_transformed.loc[:, ~prot_df_all_transformed.columns.isin(blacklist)]
rna_df_all_transformed = rna_df_all_transformed.loc[:, ~rna_df_all_transformed.columns.isin(blacklist)]
prot_df_123_transformed = prot_df_123_transformed.loc[:, ~prot_df_123_transformed.columns.isin(blacklist)]
rna_df_123_transformed = rna_df_123_transformed.loc[:, ~rna_df_123_transformed.columns.isin(blacklist)]

# check shape of all dataframes
print(f'\nexpr_df shape: {rna_df_all_transformed.shape}')
print(f'rppa_df shape: {prot_df_all_transformed.shape}')
print(f'expr_df_2_13 shape: {rna_df_123_transformed.shape}')
print(f'rppa_df_2_13 shape: {prot_df_123_transformed.shape}')

# Check for NaNs 
print(f'expr_df NaNs: {rna_df_all_transformed.isna().sum().sum()}')
print(f'rppa_df NaNs: {prot_df_all_transformed.isna().sum().sum()}')
# Check for Inf
print(f'expr_df Inf: {np.isinf(rna_df_all_transformed).sum().sum()}')
print(f'rppa_df Inf: {np.isinf(prot_df_all_transformed).sum().sum()}')

# replace NaNs with column mean
rna_df_all_transformed = rna_df_all_transformed.fillna(rna_df_all_transformed.mean())
prot_df_all_transformed = prot_df_all_transformed.fillna(prot_df_all_transformed.mean())
rna_df_123_transformed = rna_df_123_transformed.fillna(rna_df_123_transformed.mean())
prot_df_123_transformed = prot_df_123_transformed.fillsna(prot_df_123_transformed.mean())


# write to csv
prot_df_all_transformed.to_csv('../Diffusion/data/proteomics_for_pig_cmsALL.csv')
rna_df_all_transformed.to_csv('../Diffusion/data/transcriptomics_for_pig_cmsALL.csv')

prot_df_123_transformed.to_csv('../Diffusion/data/proteomics_for_pig_cms123.csv')
rna_df_123_transformed.to_csv('../Diffusion/data/transcriptomics_for_pig_cms123.csv')

# write column names to .txt file
with open('../Diffusion/data/VAR_NAMES_GENELIST.txt', 'w') as f:
    for item in prot_df_all_transformed.columns:
        f.write("%s\n" % item)


# %%
# center and scale across columns
expr_df_2 = center_and_scale(expr_df_2, axis=1)
expr_df_4 = center_and_scale(expr_df_4, axis=1)

# sum column-wise
rppa_df_2_flat = rppa_df_2.sum(axis=0)
rppa_df_4_flat = rppa_df_4.sum(axis=0)

expr_df_2_flat = expr_df_2.sum(axis=0)
expr_df_4_flat = expr_df_4.sum(axis=0)


# Create a figure and a set of subplots
fig, axs = plt.subplots(2)

# Plot bar chart for rppa_df_2_flat
axs[0].bar(expr_df_2_flat.index, expr_df_2_flat)
axs[0].set_title('df_2_flat')
axs[0].set_xticks([])  # Hide x-axis labels for clarity

# Plot bar chart for rppa_df_4_flat
axs[1].bar(expr_df_4_flat.index, expr_df_4_flat)
axs[1].set_title('df_4_flat')
axs[1].set_xticks([])  # Hide x-axis labels for clarity

# Display the plot
plt.tight_layout()
plt.show()

# Load the CSV file
pathway_df = pd.read_csv('../Diffusion/data/Pathway_Enrichment_Info.csv')

def get_gene_values(description, df_2_flat, df_4_flat):
    # Find rows where 'description' column contains the given string
    rows = pathway_df[pathway_df['description'].str.contains(description)]
    
    # Initialize dictionaries to store the results
    gene_values_2 = {}
    gene_values_4 = {}

    # Iterate over the found rows
    for _, row in rows.iterrows():
        # Split the 'genes' column into individual genes
        genes = row['genes'].split('|')
        
        # For each gene, get the value from rppa_df_2_flat and rppa_df_4_flat
        for gene in genes:
            if gene in df_2_flat.index:
                gene_values_2[gene] = df_2_flat[gene]
            if gene in df_4_flat.index:
                gene_values_4[gene] = df_4_flat[gene]

    return gene_values_2, gene_values_4

tgf_values_2, tgf_values_4 = get_gene_values('Angiogenesis', expr_df_2_flat, expr_df_4_flat)

# Initialize a list to store the subtraction results
subtraction_results = []

# Iterate over the keys in gene_values_2
for gene in tgf_values_2.keys():
    # If the gene is also in gene_values_4, subtract the values
    if gene in tgf_values_4:
        subtraction_results.append(tgf_values_4[gene] - tgf_values_2[gene])

# Calculate the percentage of positive results
percentage_positive = (sum(i > 0 for i in subtraction_results) / len(subtraction_results)) * 100

print(f'Percentage of positive values: {percentage_positive}%')

# # %%
# import numpy as np
# import scipy.stats as stats
# # Create a dataset that is a skewed normal distribution

# data = stats.skewnorm.rvs(0, size=1000)

# # skew in oposite direction
# data = -data


# # Plot the histogram
# fig, ax = plt.subplots()
# ax.hist(data, bins=50)
# ax.set_title('Skewed normal distribution')
# plt.show()

# # Perform the Shapiro-Wilk test
# stat, p = stats.shapiro(data)
# print(f'Statistic: {stat}, p-value: {p}')

# # QQ-plot
# fig, ax = plt.subplots()
# stats.probplot(data, dist="norm", plot=ax)
# ax.set_title(f'QQ-plot (W: {round(stat, 3)})')
# plt.show()















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




