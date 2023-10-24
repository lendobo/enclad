# %%
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# Load and filter the dataset
data = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_ALL_labelled.csv')
filtered_data = data[data['CMS_Label'].isin(['CMS2', 'CMS4'])]
cms2_data = filtered_data[filtered_data['CMS_Label'] == 'CMS2'].drop(columns=['Sample_ID', 'CMS_Label'])
cms4_data = filtered_data[filtered_data['CMS_Label'] == 'CMS4'].drop(columns=['Sample_ID', 'CMS_Label'])

# Perform differential expression analysis
p_values = []
adjusted_log2_fold_changes = []
for gene in cms2_data.columns:
    cms2_values = cms2_data[gene]
    cms4_values = cms4_data[gene]
    t_stat, p_val = ttest_ind(cms2_values, cms4_values, equal_var=False)
    log2_fc = cms4_values.mean() - cms2_values.mean()
    p_values.append(p_val)
    adjusted_log2_fold_changes.append(log2_fc)


# %%
# Correct p-values using the Benjamini-Hochberg procedure
corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
print(corrected_p_values)

diff_expr_results = pd.DataFrame({
    'Gene': cms2_data.columns,
    'P-Value': p_values,
    'Adjusted Log2 Fold Change': adjusted_log2_fold_changes,
    'Corrected P-Value': corrected_p_values
})

# Check for any NaN values in p_values
nan_p_values = np.isnan(p_values).sum()

print("Number of NaN p-values:", nan_p_values)
print(len(p_values))

# %%
# Generate the volcano plot
x = diff_expr_results['Adjusted Log2 Fold Change']
y_raw_p_values = -np.log10(diff_expr_results['P-Value'])
plt.figure(figsize=(10, 6))
plt.scatter(x, y_raw_p_values, s=5, c='blue', alpha=0.5)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
plt.title("Volcano Plot (using raw p-values)")
plt.xlabel("Adjusted Log2 Fold Change")
plt.ylabel("-log10(P-Value)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
# Generate the volcano plot using corrected p-values
x = diff_expr_results['Adjusted Log2 Fold Change']
y_corrected_p_values = -np.log10(diff_expr_results['Corrected P-Value'])
print(y_corrected_p_values)
plt.figure(figsize=(10, 6))
plt.scatter(x, y_corrected_p_values, s=5, c='blue', alpha=0.5)
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label="Significance threshold")
plt.title("Volcano Plot (using Benjamini-Hochberg corrected p-values)")
plt.xlabel("Adjusted Log2 Fold Change")
plt.ylabel("-log10(Corrected P-Value)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
