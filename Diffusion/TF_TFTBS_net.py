# %%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pymnet import *

# %%
tfs_and_targets = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/trrust_rawdata.human.tsv', sep='\t')
data_proteins = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/Prot_Names.csv')
data_RNA = pd.read_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/data/Synapse/TCGA/RNA_CMS_groups/RNA_names.csv')

#tf_proteins is the first column of tf_bs_proteins
tf_proteins = tfs_and_targets.iloc[:,0].unique()

# check overlap
prot_tfs = np.intersect1d(data_proteins, tf_proteins)
print(len(prot_tfs))

rna_tfs = np.intersect1d(data_RNA, prot_tfs)
print(len(rna_tfs))

# incex tf_bs_proteins at prot_tfs
tfs_and_targets = tfs_and_targets[tfs_and_targets.iloc[:,0].isin(prot_tfs)]
tf_targets_db = tfs_and_targets.iloc[:,1].unique()

targeted_RNA = np.intersect1d(tf_targets_db, data_RNA)
print(len(targeted_RNA))

targeted_PROTS = np.intersect1d(targeted_RNA, data_proteins)
print(len(targeted_PROTS))

# concatenate prot_tfs and  targeted_PROTS
prot_tfs = np.concatenate((prot_tfs, targeted_PROTS))
print(len(prot_tfs))

# write to csv
pd.DataFrame(prot_tfs).to_csv('/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/hNITR/phase_1_code/Diffusion/protein_names_diff.csv', index=False)



# %%
