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
print(cms2_data.shape)


# remove columns whose means are close to 0
cms2_data = cms2_data.loc[:, (cms2_data.mean(axis=0) > 0.1)]
cms4_data = cms4_data.loc[:, (cms4_data.mean(axis=0) > 0.1)]

# remove columns that are all 0s, NaNs or infs
cms2_data = cms2_data.loc[:, (cms2_data != 0).all(axis=0)]
cms4_data = cms4_data.loc[:, (cms4_data != 0).all(axis=0)]

# remove columns where all values are very close to 0
cms2_data = cms2_data.loc[:, (cms2_data > 0.01).any(axis=0)]
cms4_data = cms4_data.loc[:, (cms4_data != 0).any(axis=0)]

# Only keep columns which are present in both cm2 and cms4
cms2_genes = cms2_data.columns
cms4_genes = cms4_data.columns
common_genes = cms2_genes.intersection(cms4_genes)
cms2_data = cms2_data[common_genes]
cms4_data = cms4_data[common_genes]

print(cms4_data.shape)

# exponentiate 2 to the power of each sample
cms2_data_e2 = 2 ** cms2_data
cms4_data_e2 = 2 ** cms4_data

# pick 10 random genes and check for normality
random_genes = cms2_data.sample(n=10, axis=1).columns

plt.figure(figsize=(12, 10))  # Adjusted the figure size for better visualization
for i, gene in enumerate(random_genes):
    cms2_gene_values = cms2_data[gene]
    cms2_gene_values_e2 = cms2_data_e2[gene]
    
    # plot cms2_gene_values
    ax1 = plt.subplot(4, 5, i+1)  # Adjusted the grid to 4x5 and indexed correctly
    stats.probplot(cms2_gene_values, dist="norm", plot=ax1)
    
    # plot cms2_gene_values_e2
    ax2 = plt.subplot(4, 5, i+11)  # Adjusted the grid to 4x5 and indexed correctly
    stats.probplot(cms2_gene_values_e2, dist="norm", plot=ax2)
    plt.tight_layout()
plt.show()

# # remove columns whose mean values are above 50000
# cms2_data_e2 = cms2_data_e2.loc[:, (cms2_data_e2.mean(axis=0) < 20000)]
# cms4_data_e2 = cms4_data_e2.loc[:, (cms4_data_e2.mean(axis=0) < 20000)]


# # make a histogram of means
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.hist(cms2_data_e2.mean(axis=0), bins=100)
# plt.title("CMS2")
# plt.xlabel("Mean")
# # plt.xlim(0, 50000)
# plt.ylabel("Frequency")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.subplot(1, 2, 2)
# plt.hist(cms4_data_e2.mean(axis=0), bins=100)
# plt.title("CMS4")
# plt.xlabel("Mean")
# plt.ylabel("Frequency")
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

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
# LOG - DIFF FC
Log2FX_X = 0.35
# Generate the volcano plot using LOG-DIFFERENCES
x = diff_expr_results['Log-differences']
y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
plt.figure(figsize=(10, 6))
plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
plt.axvline(x=0.5, color='black', linestyle='--')
plt.axvline(x=-0.5, color='black', linestyle='--')
plt.title("Volcano Plot (using log-differences)")
plt.xlabel("Log-differences")
plt.ylabel("-log10(Corrected P-Value)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# show number of data points below -1 on the x axis
print('Log2 Diff')
print(f'Number of data points below -{Log2FX_X}: {len(diff_expr_results[diff_expr_results["Log-differences"] < -Log2FX_X])}')
print(f'Number of data points above {Log2FX_X}: {len(diff_expr_results[diff_expr_results["Log-differences"] > Log2FX_X])}')


top_n = 500

# Isolate genes that are above diff_expr_results[diff_expr_results["Log-differences"] > logX2FC and above significance threshold
significant_genes = diff_expr_results[diff_expr_results['Corrected P-Value'] < 0.05]
top_genes_cms4 = significant_genes.sort_values(by='Log-differences', ascending=False).head(top_n)
top_genes_cms2 = significant_genes.sort_values(by='Log-differences', ascending=True).head(top_n)



# index filtered_data[filtered_data['CMS_Label'] == 'CMS4'] at significant genes
filtered_cms4_data = filtered_data[filtered_data['CMS_Label'] == 'CMS4'][top_genes_cms4.index]
filtered_cms4_data.to_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_top_DEA.csv')
# write names to file
with open(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/top{top_n}_cms4_names.txt', 'w') as f:
    for gene in top_genes_cms4.index:
        f.write(gene + '\n')

# same for CMS2
filtered_cms2_data = filtered_data[filtered_data['CMS_Label'] == 'CMS2'][top_genes_cms2.index]
filtered_cms2_data.to_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms2_top_DEA.csv')
with open(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/top{top_n}_cms2_names.txt', 'w') as f:
    for gene in top_genes_cms2.index:
        f.write(gene + '\n')

# Write a transposed version to a tab-separated file
filtered_cms4_data.T.to_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_top{top_n}_transposed.tsv', sep='\t')
filtered_cms2_data.T.to_csv(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms2_top{top_n}_transposed.tsv', sep='\t')


# %%
# logX2FC = 0.5
# aboveFC = 1 / logX2FC
# belowFC = 1 * logX2FC

# # Generate the volcano plot using corrected p-values
# x = diff_expr_results['Adjusted Log2 Fold Change']
# y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
# print(y_corrected_p_values)
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
# plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
# plt.axvline(x=aboveFC, color='black', linestyle='--')
# plt.axvline(x=belowFC, color='black', linestyle='--')
# plt.title("Volcano Plot (using Benjamini-Hochberg corrected p-values)")
# plt.xlabel("Adjusted Log2 Fold Change")
# plt.xlim(0,5)
# plt.ylabel("-log10(Corrected P-Value)")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# # show number of data points below -1 on the x axis
# print('Log2FC')
# print(f'Number of data points below {belowFC}: {len(diff_expr_results[diff_expr_results["Adjusted Log2 Fold Change"] < belowFC])}')
# print(f'Number of data points above {aboveFC}: {len(diff_expr_results[diff_expr_results["Adjusted Log2 Fold Change"] > aboveFC])}')

# # of the significant genes, select the top 500 with highest fold change
# significant_genes = diff_expr_results[diff_expr_results['Corrected P-Value'] < 0.05]
# significant_genes = significant_genes.sort_values(by='Adjusted Log2 Fold Change', ascending=False).head(500)

# # Filter cms4 dataset by these genes and write to csv
# filtered_cms4_data = filtered_cms4_data[significant_genes.index]
# filtered_cms4_data.to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_top500.csv')


# %%
# # Generate the volcano plot using rank-based log2 fold changes
# x = diff_expr_results['Rank-Based Log2 Fold Change']
# y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
# plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
# plt.axvline(x=1, color='black', linestyle='--')
# plt.axvline(x=-1, color='black', linestyle='--')
# plt.title("Volcano Plot (using rank-based log2 fold changes)")
# plt.xlabel("Rank-Based Log2 Fold Change")
# plt.ylabel("-log10(Corrected P-Value)")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# logX2FC = 0.5
# # show number of data points below -1 on the x axis
# print('Rank-based LOg2FC')
# print(f'Number of data points below {logX2FC}: {len(diff_expr_results[diff_expr_results["Rank-Based Log2 Fold Change"] < -logX2FC])}')
# print(f'Number of data points above {logX2FC}: {len(diff_expr_results[diff_expr_results["Rank-Based Log2 Fold Change"] > logX2FC])}')




# %%

# window = 500
# # print top 50 CMS4 genes
# top_cms4_genes = diff_expr_results.sort_values(by='Adjusted Log2 Fold Change', ascending=False).head(window)
# print(top_cms4_genes.index)
# # top 50 rank based
# top_rank_based_genes = diff_expr_results.sort_values(by='Rank-Based Log2 Fold Change', ascending=False).head(window)
# top_rank_based_gene_names = list(top_rank_based_genes.index)
# # top 50 log-differences
# top_logdiff_genes = diff_expr_results.sort_values(by='Log-differences', ascending=False).head(window)
# top_logdiff_gene_names = list(top_logdiff_genes.index)

# # Write all genes that are above 0 for adjusted log2 fold change to file
# cms4_up_genes = diff_expr_results[diff_expr_results['Adjusted Log2 Fold Change'] > 0].index
# print (f'Number of genes upregulated in CMS4: {len(cms4_up_genes)}')
# with open('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_cms4_up.csv', 'w') as f:
#     for gene in cms4_up_genes:
#         f.write(gene + '\n')

# # check for number of genes in common between all three
# common_log2FC_rank = len(set(top_cms4_genes.index) & set(top_rank_based_genes.index)) / window
# common_log2FC_logdiff = len(set(top_cms4_genes.index) & set(top_logdiff_genes.index)) / window
# common_rank_logdiff = len(set(top_rank_based_genes.index) & set(top_logdiff_genes.index)) / window

# print(f'Common genes between log2FC and rank-based (%): {common_log2FC_rank}')
# print(f'Common genes between log2FC and log-differences (%): {common_log2FC_logdiff}')
# print(f'Common genes between rank-based and log-differences (%): {common_rank_logdiff}')

# # %% 
# # write gene names of log-differences to file
# with open(f'/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_top{window}_rankbased_gene_names.txt', 'w') as f:
#     for gene in top_rank_based_gene_names:
#         f.write(gene + '\n')
# # %%
# # # LOAD ORPHAN NODES
# # orphan_nodes = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/orphan_cms4.csv')
# # orphan_nodes = orphan_nodes.set_index('Gene')

# # compare number of orphan nodes with top{window} log-differences
# common_orphan_logdiff = len(set(top_logdiff_genes.index) & set(orphan_nodes.index)) / window
# print(f'Common genes between log-differences and orphan nodes (%): {common_orphan_logdiff}')

# # %%
