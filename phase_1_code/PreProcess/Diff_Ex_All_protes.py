
# %%
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# Defining the file paths for the uploaded protein abundance matrices
file_paths = {
    'CMS1': 'data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS1.csv',
    'CMS2': 'data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS2.csv',
    'CMS3': 'data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS3.csv',
    'CMS4': 'data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS4.csv',
}

# Loading the entire datasets
data_full = {}
missing_values_info = {}

for subtype, file_path in file_paths.items():
    # Load the data
    data_full[subtype] = pd.read_csv(file_path, index_col=0)
    
    # Check for missing values
    missing_values_info[subtype] = data_full[subtype].isna().sum().sum()

missing_values_info

# %%
# Initialize a dictionary to store t-test results
ttest_results = {}

# Loop through each protein to perform t-test
for protein in protein_list:
    # Extract abundance levels for this protein for CMS2 and CMS4
    cms2_abundance = data_full['CMS2'].loc[protein].values
    cms4_abundance = data_full['CMS4'].loc[protein].values
    
    # Perform t-test
    t_stat, p_value = ttest_ind(cms2_abundance, cms4_abundance, equal_var=False)
    
    # Store the results
    ttest_results[protein] = {'t-statistic': t_stat, 'p-value': p_value}

# Convert the results to a DataFrame for easier manipulation and viewing
ttest_df = pd.DataFrame.from_dict(ttest_results, orient='index')

# Perform multiple hypothesis testing correction using Benjamini-Hochberg procedure
corrected_p_values_ttest = multipletests(ttest_df['p-value'], method='fdr_bh')[1]

# Add the corrected p-values to the DataFrame
ttest_df['corrected_p-value'] = corrected_p_values_ttest

# Sort the DataFrame based on corrected p-values to see the most significant proteins first
sorted_ttest_df = ttest_df.sort_values(by='corrected_p-value')

# Identify proteins that are significantly differentially abundant
alpha = 0.05
significant_proteins_ttest = sorted_ttest_df[sorted_ttest_df['corrected_p-value'] <= alpha]

# Save the list of significant proteins to a CSV file
significant_file_path_ttest = '/mnt/data/Significant_Differentially_Abundant_Proteins_CMS2_vs_CMS4.csv'
significant_proteins_ttest.to_csv(significant_file_path_ttest)

significant_file_path_ttest, significant_proteins_ttest.head()

# %%
