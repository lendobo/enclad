
# %%
import pandas as pd
import numpy as np



protein_list_L3 = pd.read_csv('../data/TCGA-COAD-L3-S42/tmp/TCGA-COAD-L3-S42.csv', index_col=0)
protein_list_L4 = pd.read_csv('../data/TCGA-COAD-L4/tmp/TCGA-COAD-L4.csv', index_col=0)

protein_list_L3.head()
protein_list_L4.head()

# %%
# select column 'peptide target'
protein_names = protein_list_L4.columns

RNA_list = pd.read_csv('../data/Synapse/TCGA/RNA_CMS_groups/RNA_names.csv')
RNA_list = RNA_list.loc[:, ['Sample_ID']].astype(str)

# print(protein_names)

# Find the overlapping names
overlap = protein_names.intersection(RNA_list['Sample_ID'])

print(len(overlap))
