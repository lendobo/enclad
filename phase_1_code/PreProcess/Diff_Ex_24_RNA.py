# %%
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import statsmodels.api as sm

# Load and filter the dataset
data = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_ALL_labelled.csv')
filtered_data = data[data['CMS_Label'].isin(['CMS2', 'CMS4'])]
cms2_data = filtered_data[filtered_data['CMS_Label'] == 'CMS2'].drop(columns=['Sample_ID', 'CMS_Label'])
cms4_data = filtered_data[filtered_data['CMS_Label'] == 'CMS4'].drop(columns=['Sample_ID', 'CMS_Label'])

# create plot of mean vs SD for each dataset
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(cms2_data.mean(axis=0), cms2_data.std(axis=0), s=5, c='blue', alpha=0.5, label='CMS2')
plt.title("CMS2")
plt.xlabel("Mean")
plt.ylabel("Standard Deviation")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.subplot(1, 2, 2)
plt.scatter(cms4_data.mean(axis=0), cms4_data.std(axis=0), s=5, c='red', alpha=0.5, label='CMS4')
plt.title("CMS4")
plt.xlabel("Mean")
plt.ylabel("Standard Deviation")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# # exponentiate 10 to the power of each sample
cms2_data_e10 = 10 ** cms2_data
cms4_data_e10 = 10 ** cms4_data 

# exponentiate 2 to the power of each sample
cms2_data_e2 = 2 ** cms2_data
cms4_data_e2 = 2 ** cms4_data

# print largest 10 values in each dataset
print(cms2_data.max().nlargest(10))
print(cms4_data.max().nlargest(10))


# # remove values that are greater than 100000
# cms2_data_e2 = cms2_data_e2[cms2_data_e2 < 100000]
# cms4_data_e2 = cms4_data_e2[cms4_data_e2 < 100000]

# # remove values that are greater than 100000
# cms2_data_e10 = cms2_data_e10[cms2_data_e10 < 100000]
# cms4_data_e10 = cms4_data_e10[cms4_data_e10 < 100000]

# # create a multiplot that compares e10 and e2 for cms2 and cms4
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(np.log10(cms2_data_e10.mean(axis=0)), np.log10(cms2_data_e10.std(axis=0)), s=5, c='red', alpha=0.5, label='CMS2')
# plt.scatter(np.log2(cms2_data_e2.mean(axis=0)), np.log2(cms2_data_e2.std(axis=0)), s=5, c='blue', alpha=0.5, label='CMS2')
# plt.title("e10")
# plt.xlabel("Mean")
# plt.ylabel("Standard Deviation")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.subplot(1, 2, 2)
# plt.scatter(np.log2(cms4_data_e2.mean(axis=0)), np.log2(cms4_data_e2.std(axis=0)), s=5, c='blue', alpha=0.5, label='CMS4')
# plt.scatter(np.log10(cms4_data_e10.mean(axis=0)), np.log10(cms4_data_e10.std(axis=0)), s=5, c='red', alpha=0.5, label='CMS4')
# plt.title("e2")
# plt.xlabel("Mean")
# plt.ylabel("Standard Deviation")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# remove columns that are all 0s, NaNs or infs
# cms2_data = cms2_data.loc[:, (cms2_data != 0).any(axis=0)]
# remove columns where all values are very close to 0
# cms2_data = cms2_data.loc[:, (cms2_data > 0.01).any(axis=0)]

# # Extract expression levels for the gene "A1BG"
# cms2_gene_values = cms2_data["A1BG"]
# cms4_gene_values = cms4_data["A1BG"]

# random_genes = cms2_data.sample(n=1, axis=1).columns

# # Plot Q-Q plot for cms2_data
# plt.figure(figsize=(12, 5))
# # stats.probplot(cms2_gene_values, dist="norm", plot=plt)
# plt.title("Q-Q plot for A1BG in cms2_data")
# # Plot Q-Q plot for cms4_data

# for gene in random_genes:
#     cms2_gene_values = cms2_data[gene]
#     cms4_gene_values = cms4_data[gene]
#     plt.subplot(1, 2, 1)
#     stats.probplot(cms2_gene_values, dist="norm", plot=plt)
#     plt.subplot(1, 2, 2)
#     stats.probplot(cms4_gene_values, dist="norm", plot=plt)
# plt.title("Q-Q plot for A1BG in cms4_data")
# plt.tight_layout()
# plt.show()

# %%

# Calculate means and standard deviations for cms2 and cms4
cms2_mean = cms2_data.mean(axis=0)
cms2_sd = cms2_data.std(axis=0)

cms4_mean = cms4_data.mean(axis=0)
cms4_sd = cms4_data.std(axis=0)

# # scale and center the data
cms2_data = (cms2_data - cms2_data.mean()) / cms2_data.std()
cms4_data = (cms4_data - cms4_data.mean()) / cms4_data.std()

cms2_data.head()

# # CHecck for NaNs or infs
# nan_or_inf_in_mean = np.isnan(cms2_mean) | np.isinf(cms2_mean)
# nan_or_inf_in_sd = np.isnan(cms2_sd) | np.isinf(cms2_sd)

# problematic_genes = cms2_mean.index[nan_or_inf_in_mean | nan_or_inf_in_sd]
# print(len(problematic_genes))

# # remove problematic genes from dataset
# cms2_data = cms2_data.drop(columns=problematic_genes)


# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(np.log10(cms2_mean), np.log10(cms2_sd), s=5, c='blue', alpha=0.5, label='CMS2')
plt.scatter(np.log10(cms4_mean), np.log10(cms4_sd), s=5, c='red', alpha=0.5, label='CMS4')
plt.title("Original Data")
plt.xlabel("Mean")
plt.ylabel("Standard Deviation")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# # Plot the transformed data
# plt.figure(figsize=(10, 6))
# plt.scatter(transformed_cms2_mean, transformed_cms2_sd, s=5, c='blue', alpha=0.5, label='CMS2')
# plt.scatter(transformed_cms4_mean, transformed_cms4_sd, s=5, c='red', alpha=0.5, label='CMS4')
# plt.title("Transformed Data")
# plt.xlabel("Transformed Mean")
# plt.ylabel("Transformed Standard Deviation")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# %%

# plot sample standard deviations against sample means on a logarithmic scale
plt.figure(figsize=(10, 6))
plt.scatter(np.log10(cms2_data.mean(axis=0)), np.log10(cms2_data.std(axis=0)), s=5, c='blue', alpha=0.5, label='CMS2')
plt.scatter(np.log10(cms4_data.mean(axis=0)), np.log10(cms4_data.std(axis=0)), s=5, c='red', alpha=0.5, label='CMS4')
plt.title("Log10 Sample Standard Deviations vs. Log10 Sample Means")
plt.xlabel("Log10 Sample Mean")
plt.ylabel("Log10 Sample Standard Deviation")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Identify genes with constant expression values across all samples
constant_genes = filtered_data.drop(columns=['Sample_ID', 'CMS_Label']).nunique(axis=0)
constant_genes = constant_genes[constant_genes == 1].index

# Exclude these genes from the dataset
filtered_cms2_data = cms2_data.drop(columns=constant_genes, errors='ignore')
filtered_cms4_data = cms4_data.drop(columns=constant_genes, errors='ignore')

# # load file from Giant Component to dataframe
# giant_component_genes = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/Cytoscape_FIViz/RNA_FI/GC_nodes_RNA.csv')

# # filter out genes that are not in giant component
# filtered_cms2_data = filtered_cms2_data[giant_component_genes['Gene']]
# filtered_cms4_data = filtered_cms4_data[giant_component_genes['Gene']]

# # write all gene names to file
# with open('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_all_gene_names.txt', 'w') as f:
#     for gene in filtered_cms2_data.columns:
#         f.write(gene + '\n')

# %%
# Differential Expression ANalysis
# Determine the constant to add based on the smallest value in the dataset
constant_to_add = abs(filtered_data.drop(columns=['Sample_ID', 'CMS_Label']).min().min()) + 0.01

# Perform differential expression analysis
p_values = []
adjusted_log2_fold_changes = []
logdiffs = []
for gene in filtered_cms2_data.columns:
    cms2_values = filtered_cms2_data[gene]
    cms4_values = filtered_cms4_data[gene]
    t_stat, p_val = ttest_ind(cms2_values, cms4_values, equal_var=True)

    # calculate fold change as difference of means
    logdiff = cms4_values.mean() - cms2_values.mean()
    logdiffs.append(logdiff)

    # calculate fold change as ratio of means
    mean_cms2_shifted = np.mean(cms2_values) #+ constant_to_add NO LONGER SHIFTED
    mean_cms4_shifted = np.mean(cms4_values) #+ constant_to_add
    log2_fc = mean_cms4_shifted / mean_cms2_shifted # np.log2(mean_cms4_shifted / mean_cms2_shifted)

    p_values.append(p_val)
    adjusted_log2_fold_changes.append(log2_fc)

# # Correct p-values using the Benjamini-Hochberg procedure
corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
# print(f'p-values: {p_values}')
# print(f'corrected ps: {corrected_p_values}')


diff_expr_results = pd.DataFrame({
    'Gene': filtered_cms2_data.columns,
    'P-Value': p_values,
    'Adjusted Log2 Fold Change': adjusted_log2_fold_changes,
    'Log-differences': logdiffs,
    'Corrected P-Value': corrected_p_values,

})

# Compute ranks for each gene within each group
cms2_ranks = filtered_cms2_data.rank(axis=1, method='average').mean(axis=0)
cms4_ranks = filtered_cms4_data.rank(axis=1, method='average').mean(axis=0)

# Compute the log2 fold change based on ratios of ranks
rank_based_log2_fc = (cms4_ranks + 1) / (cms2_ranks + 1) # np.log2((cms4_ranks + 1) / (cms2_ranks + 1))

# Update the fold changes in the results DataFrame
diff_expr_results.set_index('Gene', inplace=True)
diff_expr_results['Rank-Based Log2 Fold Change'] = rank_based_log2_fc

# %%
logX2FC = 0.5
aboveFC = 1 / logX2FC
belowFC = 1 * logX2FC

# Generate the volcano plot using corrected p-values
x = diff_expr_results['Adjusted Log2 Fold Change']
y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
print(y_corrected_p_values)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
plt.axvline(x=aboveFC, color='black', linestyle='--')
plt.axvline(x=belowFC, color='black', linestyle='--')
plt.title("Volcano Plot (using Benjamini-Hochberg corrected p-values)")
plt.xlabel("Adjusted Log2 Fold Change")
plt.xlim(0,5)
plt.ylabel("-log10(Corrected P-Value)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# show number of data points below -1 on the x axis
print(f'Number of data points below {belowFC}: {len(diff_expr_results[diff_expr_results["Adjusted Log2 Fold Change"] < belowFC])}')
print(f'Number of data points above {aboveFC}: {len(diff_expr_results[diff_expr_results["Adjusted Log2 Fold Change"] > aboveFC])}')

# of the significant genes, select the top 500 with highest fold change
significant_genes = diff_expr_results[diff_expr_results['Corrected P-Value'] < 0.05]
significant_genes = significant_genes.sort_values(by='Adjusted Log2 Fold Change', ascending=False).head(500)

# Filter cms4 dataset by these genes and write to csv
filtered_cms4_data = filtered_cms4_data[significant_genes.index]
filtered_cms4_data.to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_top500.csv')

# print mean value of FAM123A gene for cms2 and cms4
print(f'CMS2 mean: {filtered_cms2_data["FAM123A"].mean()}')
print(f'CMS4 mean: {filtered_cms4_data["FAM123A"].mean()}')

# %%
# Generate the volcano plot using rank-based log2 fold changes
x = diff_expr_results['Rank-Based Log2 Fold Change']
y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
plt.figure(figsize=(10, 6))
plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
plt.axvline(x=1, color='black', linestyle='--')
plt.axvline(x=-1, color='black', linestyle='--')
plt.title("Volcano Plot (using rank-based log2 fold changes)")
plt.xlabel("Rank-Based Log2 Fold Change")
plt.ylabel("-log10(Corrected P-Value)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

logX2FC = 0.5
# show number of data points below -1 on the x axis
print(f'Number of data points below {logX2FC}: {len(diff_expr_results[diff_expr_results["Rank-Based Log2 Fold Change"] < logX2FC])}')
print(f'Number of data points above {logX2FC}: {len(diff_expr_results[diff_expr_results["Rank-Based Log2 Fold Change"] > logX2FC])}')

# %%
# Generate the volcano plot using log-differences
x = diff_expr_results['Log-differences']
y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
plt.figure(figsize=(10, 6))
plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
plt.axvline(x=1, color='black', linestyle='--')
plt.axvline(x=-1, color='black', linestyle='--')
plt.title("Volcano Plot (using log-differences)")
plt.xlabel("Log-differences")
plt.ylabel("-log10(Corrected P-Value)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

logX2FC = 0.5
# show number of data points below -1 on the x axis
print(f'Number of data points below -{logX2FC}: {len(diff_expr_results[diff_expr_results["Log-differences"] < logX2FC])}')
print(f'Number of data points above {logX2FC}: {len(diff_expr_results[diff_expr_results["Log-differences"] > logX2FC])}')

# %%

window = 500
# print top 50 CMS4 genes
top_cms4_genes = diff_expr_results.sort_values(by='Adjusted Log2 Fold Change', ascending=False).head(window)
print(top_cms4_genes.index)
# top 50 rank based
top_rank_based_genes = diff_expr_results.sort_values(by='Rank-Based Log2 Fold Change', ascending=False).head(window)
top_rank_based_gene_names = list(top_rank_based_genes.index)
# top 50 log-differences
top_logdiff_genes = diff_expr_results.sort_values(by='Log-differences', ascending=False).head(window)
top_logdiff_gene_names = list(top_logdiff_genes.index)

# Write all genes that are above 0 for adjusted log2 fold change to file
cms4_up_genes = diff_expr_results[diff_expr_results['Adjusted Log2 Fold Change'] > 0].index
print (f'Number of genes upregulated in CMS4: {len(cms4_up_genes)}')
with open('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_up.csv', 'w') as f:
    for gene in cms4_up_genes:
        f.write(gene + '\n')

# check for number of genes in common between all three
common_log2FC_rank = len(set(top_cms4_genes.index) & set(top_rank_based_genes.index)) / window
common_log2FC_logdiff = len(set(top_cms4_genes.index) & set(top_logdiff_genes.index)) / window
common_rank_logdiff = len(set(top_rank_based_genes.index) & set(top_logdiff_genes.index)) / window

print(f'Common genes between log2FC and rank-based (%): {common_log2FC_rank}')
print(f'Common genes between log2FC and log-differences (%): {common_log2FC_logdiff}')
print(f'Common genes between rank-based and log-differences (%): {common_rank_logdiff}')

# %% 
# write gene names of log-differences to file
with open(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_top{window}_rankbased_gene_names.txt', 'w') as f:
    for gene in top_rank_based_gene_names:
        f.write(gene + '\n')
# %%
# # LOAD ORPHAN NODES
# orphan_nodes = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/orphan_cms4.csv')
# orphan_nodes = orphan_nodes.set_index('Gene')

# compare number of orphan nodes with top{window} log-differences
common_orphan_logdiff = len(set(top_logdiff_genes.index) & set(orphan_nodes.index)) / window
print(f'Common genes between log-differences and orphan nodes (%): {common_orphan_logdiff}')

# %%
