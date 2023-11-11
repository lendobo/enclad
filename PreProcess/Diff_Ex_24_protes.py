# Fully Consolidated Code Block including Data Loading, Analysis, and Plotting

# Import required libraries
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the protein abundance matrices for the two conditions
cms2_data = pd.read_csv('../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS2.csv')
cms4_data = pd.read_csv('../data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS4.csv')

# Set the index to protein names for easier manipulation
cms2_data.set_index('Unnamed: 0', inplace=True)
cms4_data.set_index('Unnamed: 0', inplace=True)

# Initialize empty dictionaries to hold fold changes and p-values
fold_changes_full = {}
p_values_full = {}

# Iterate over each protein and calculate fold change and p-value
for protein in cms2_data.index:
    if protein in cms4_data.index:
        cms2_protein_data = cms2_data.loc[protein]
        cms4_protein_data = cms4_data.loc[protein]
        
        # Calculate mean abundance in each condition
        mean_cms2 = np.mean(cms2_protein_data)
        mean_cms4 = np.mean(cms4_protein_data)
        
        # Calculate fold change as ratio of means
        if mean_cms2 == 0 and mean_cms4 == 0:
            fold_change = np.nan
        elif mean_cms2 == 0:
            fold_change = np.inf
        elif mean_cms4 == 0:
            fold_change = -np.inf
        else:
            fold_change = np.log2(mean_cms4 / mean_cms2)
        
        # Perform t-test to get p-value
        t_stat, p_value = ttest_ind(cms2_protein_data, cms4_protein_data)
        
        # Store the results
        fold_changes_full[protein] = fold_change
        p_values_full[protein] = p_value

# Create a DataFrame to hold the calculated values and remove NaNs
results_df_full = pd.DataFrame({
    'Fold Change': fold_changes_full,
    'P-Value': p_values_full
}).dropna()

# Store proteins with infinite fold changes in a separate list
proteins_uniquely_in_cms4_full = results_df_full[results_df_full['Fold Change'].apply(np.isinf)].index.tolist()

# Filter out the infinite values for plotting
filtered_results_df_full = results_df_full.replace([np.inf, -np.inf], np.nan).dropna()

# Calculate the negative log10 of the p-values
filtered_results_df_full['-log10(P-Value)'] = -np.log10(filtered_results_df_full['P-Value'])

# Generate the Volcano Plot
plt.figure(figsize=(14, 10))
sns.scatterplot(x='Fold Change', y='-log10(P-Value)', data=filtered_results_df_full,
                hue='Fold Change', palette="coolwarm_r",
                hue_norm=(-max(abs(filtered_results_df_full['Fold Change'])), max(abs(filtered_results_df_full['Fold Change']))))
plt.title('Volcano Plot of Protein Abundance between CMS2 and CMS4 (Filtered)', fontsize=16)
plt.xlabel('Log2(Fold Change)', fontsize=14)
plt.ylabel('-Log10(P-Value)', fontsize=14)
plt.axhline(y=-np.log10(0.05), linestyle='--', color='black')
plt.text(max(filtered_results_df_full['Fold Change'])-1, -np.log10(0.05)+0.2, 'p-value=0.05', fontsize=12)
plt.show()

proteins_uniquely_in_cms4_full, filtered_results_df_full.head()
