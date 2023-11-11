# %%
import pandas as pd

# File paths
rna_file_path = "../data/Synapse/TCGA/TCGACRC_expression.tsv"
label_file_path = "../data/Synapse/mergedPhenotype/cms_labels_public_all.txt"

# Read the RNA data (TSV format) and set "feature" as the index
rna_data = pd.read_csv(rna_file_path, sep="\t")
rna_data.set_index("feature", inplace=True)

# Read and filter the label data
label_data_filtered = pd.read_csv(label_file_path, header=None, skiprows=range(0, 2826), usecols=[0, 4], sep="\t")
label_data_filtered.columns = ["Sample_ID", "CMS_Label"]
label_data_filtered = label_data_filtered[label_data_filtered['Sample_ID'].str.startswith('TCGA-')]
label_data_filtered['Sample_ID'] = label_data_filtered['Sample_ID'].apply(lambda x: '-'.join(x.split('-')[0:3]))

# Prepare the RNA data for merging
rna_data_for_merge = rna_data.transpose()
rna_data_for_merge['Sample_ID'] = rna_data_for_merge.index
rna_data_for_merge['Sample_ID'] = rna_data_for_merge['Sample_ID'].apply(lambda x: '-'.join(x.split('-')[0:3]))

# Merge the label data with the RNA data
merged_data = pd.merge(rna_data_for_merge, label_data_filtered, on="Sample_ID", how="left")

# Splitting the data based on CMS labels
cms1_data = merged_data[merged_data['CMS_Label'] == 'CMS1'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()
cms2_data = merged_data[merged_data['CMS_Label'] == 'CMS2'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()
cms3_data = merged_data[merged_data['CMS_Label'] == 'CMS3'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()
cms4_data = merged_data[merged_data['CMS_Label'] == 'CMS4'].set_index('Sample_ID').drop(columns=['CMS_Label']).transpose()

# %%
# Place column 'CMS_label as the first column in merged data
merged_data = merged_data[['CMS_Label'] + [col for col in merged_data.columns if col != 'CMS_Label']]
# Combined labelled file
merged_data.set_index('Sample_ID').to_csv("../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_ALL_labelled.csv")

merged_data.set_index('Sample_ID').head()

# %%
# Save the new dataframes as separate .csv files
cms1_data.to_csv("../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_CMS1.csv")
cms2_data.to_csv("../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_CMS2.csv")
cms3_data.to_csv("../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_CMS3.csv")
cms4_data.to_csv("../data/Synapse/TCGA/RNA_CMS_groups/TCGACRC_expression_CMS4.csv")

# Drop columns where CMS labels are NaN
filtered_data = merged_data.transpose().dropna(subset=['CMS_Label'], axis=1)
filtered_data = filtered_data.loc[:, ~filtered_data.loc['CMS_Label'].isin(['NOLBL'])]

# Separate the CMS labels and the RNA expression data
cms_labels_filtered = filtered_data.loc['CMS_Label']
rna_data_filtered = filtered_data.drop('CMS_Label').apply(pd.to_numeric, errors='coerce')

# Write both to files
cms_labels_filtered.to_csv("../data/Synapse/TCGA/RNA_CMS_groups/labels_rna.csv")
rna_data_filtered.to_csv("../data/Synapse/TCGA/RNA_CMS_groups/rna_only.csv")


# %%
